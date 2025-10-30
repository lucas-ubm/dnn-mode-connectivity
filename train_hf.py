import argparse
import os
import sys
import json
import tabulate
import time
import torch
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from torch.cuda.amp import autocast, GradScaler
from datasets import Dataset
from evaluate import load
import numpy as np

import curves
import data
import models
import utils
import losses


def load_loss_config(config_path):
    """Load loss configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def convert_to_hf_dataset(loader):
    """Convert PyTorch DataLoader to HuggingFace Dataset"""
    all_data = []
    all_labels = []
    
    for inputs, targets in loader:
        all_data.append(inputs.numpy())
        all_labels.append(targets.numpy())
    
    return Dataset.from_dict({
        'pixel_values': np.concatenate(all_data),
        'labels': np.concatenate(all_labels)
    })


def compute_metrics(eval_pred, loss_tracker, metrics:list=None):
    """Compute metrics using HuggingFace's evaluate library"""
    predictions, labels = eval_pred
    # Ensure predictions are float32 and labels are long
    
    results = {}
    
    # Compute accuracy
    accuracy_metric = load("accuracy")
    results["accuracy"] = accuracy_metric.compute(predictions=predictions.argmax(-1), references=labels)["accuracy"]
    
    # Compute F1 score
    f1_metric = load("f1")
    results["f1"] = f1_metric.compute(predictions=predictions.argmax(-1), references=labels, average="weighted")["f1"]

    # Balanced accuracy makes no sense because of equal class balance in CIFAR-10 

    # cm_metric = load("confusion_matrix")
    # cm = cm_metric.compute(predictions=predictions.argmax(-1), references=labels, normalize= "true")["confusion_matrix"]
    # balanced_accuracy = np.diag(cm).mean()
    # results["balanced_accuracy"] = balanced_accuracy
    # recall_per_class = np.diag(cm)
    # print("Recall per class:", recall_per_class)
    # print("Normalized by true labels check", cm.sum(axis=1))
    # print("Balanced Accuracy:", recall_per_class.mean())
    # print("Accuracy:", (predictions.argmax(-1) == labels).mean())
    
    # Compute ROC AUC

    if predictions.shape[1] == 2:
        roc_metric = load("roc_auc")

        # Binary classification
        results["roc_auc"] = roc_metric.compute(
            references=labels,
            prediction_scores=predictions,  # use probabilities for both classes
        )["roc_auc"]
    else:
        # Multi-class classification
        roc_metric = load("roc_auc", "multiclass")
        refs = labels.astype(np.int32)
        preds = F.softmax(torch.from_numpy(predictions).float(), dim=1).cpu().numpy()  # shape (N, C)

        results["roc_auc"] = roc_metric.compute(
            references=refs,
            prediction_scores=preds,  # shape (N, num_classes)
            multi_class='ovr'               # one-vs-rest for multi-class
        )["roc_auc"]

    
    # Compute losses
    preds_tensor = torch.from_numpy(predictions).float()
    labels_tensor = torch.from_numpy(labels).long()
    results["loss"] = loss_tracker.main_loss(preds_tensor, labels_tensor).item()
    for name, aux_loss in loss_tracker.auxiliary_losses.items():
        results[name] = aux_loss(preds_tensor, labels_tensor).item()    
    return results


def train_epoch(model, loader, optimizer, scaler, loss_tracker, regularizer=None, verbose=False):
    """Train for one epoch with HuggingFace metrics"""
    model.train()
    
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    for input, target in loader:
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            output = model(input)
            loss = loss_tracker.main_loss(output, target)

            if regularizer is not None:
                loss = loss + regularizer(model)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        predictions = F.softmax(output, dim=1).detach().cpu().numpy()
        labels = target.detach().cpu().numpy()
        
        all_predictions.append(predictions)
        all_labels.append(labels)
        total_loss += loss.item()
        num_batches += 1

    # Compute metrics using HuggingFace's evaluate
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    metrics = compute_metrics((all_predictions, all_labels), loss_tracker)
    metrics["loss"] = total_loss / num_batches
    
    return metrics


@torch.no_grad()
def test(model, loader, loss_tracker, regularizer=None, verbose=False):
    """Evaluate model with HuggingFace metrics"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    for input, target in loader:
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with autocast():
            output = model(input)
            loss = loss_tracker.main_loss(output, target)

            if regularizer is not None:
                loss = loss + regularizer(model)

        predictions = F.softmax(output, dim=1).detach().cpu().numpy()
        labels = target.detach().cpu().numpy()
        
        all_predictions.append(predictions)
        all_labels.append(labels)
        total_loss += loss.item()
        num_batches += 1

    # Compute metrics using HuggingFace's evaluate
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    metrics = compute_metrics((all_predictions, all_labels), loss_tracker)
    metrics["loss"] = total_loss / num_batches
    
    return metrics


def setup_loss_tracker(args):
    """Setup loss tracker based on configuration"""
    if args.loss_config:
        # Load loss from config file
        loss_config = load_loss_config(args.loss_config)
        main_loss = losses.get_loss(loss_config['main_loss']['type'],
                                  **loss_config['main_loss'].get('params', {}))
        auxiliary_losses = {}
        for i, loss_cfg in enumerate(loss_config.get('auxiliary_losses', [])):
            aux_loss = losses.get_loss(loss_cfg['type'],
                                     name=f'aux_{loss_cfg["type"]}_{i}',
                                     **loss_cfg.get('params', {}))
            auxiliary_losses[aux_loss.name] = aux_loss
    elif args.loss_spec:
        # Use directly specified loss
        main_loss = losses.get_loss(args.loss_spec['type'],
                                  **args.loss_spec.get('params', {}))
        auxiliary_losses = {}
    else:
        # Default to cross entropy
        main_loss = losses.CrossEntropyLoss()
        auxiliary_losses = {}
    
    return losses.LossTracker(main_loss, auxiliary_losses)

def validate_metrics(computed_metrics, requirements):
    """Validate computed metrics against requirements"""
    if not requirements:
        return True
    
    metrics = requirements['metrics']
    thresholds = requirements['thresholds']
    
    for metric, threshold in zip(metrics, thresholds):
        if computed_metrics[f'test_{metric}'] < threshold:
            return False
    return True

def get_args(args=None):
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser(description='DNN curve training with HuggingFace metrics')
    parser.add_argument('--model', type=str, metavar='MODEL', help='model name (default: None)')
    parser.add_argument('--experiment_name', type=str, metavar='ENAME', help='name of the experiment being carried out')
    parser.add_argument('--dir', type=str, metavar='DIR', help='training directory (default: /tmp/curve/)')
    parser.add_argument('--loss_config', type=str, help='path to loss configuration file')
    parser.add_argument('--dataset', type=str, metavar='DATASET', help='dataset name (default: CIFAR10)')
    parser.add_argument('--loss_spec', type=dict, help='direct loss specification')
    parser.add_argument('--use_test', action='store_true', help='switches between validation and test set (default: test)')
    parser.add_argument('--transform', type=str, metavar='TRANSFORM', help='transform name (default: VGG)')
    parser.add_argument('--data_path', type=str, metavar='PATH', help='path to datasets location (default: None)')
    parser.add_argument('--batch_size', type=int, metavar='N', help='input batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, metavar='N', help='number of workers (default: 4)')
    parser.add_argument('--curve', type=str, metavar='CURVE', help='curve type to use (default: None)')
    parser.add_argument('--num_bends', type=int, metavar='N', help='number of curve bends (default: 3)')
    parser.add_argument('--init_start', type=str, metavar='CKPT', help='checkpoint to init start point (default: None)')
    parser.add_argument('--fix_start', dest='fix_start', action='store_true', help='fix start point (default: off)')
    parser.add_argument('--init_end', type=str, metavar='CKPT', help='checkpoint to init end point (default: None)')
    parser.add_argument('--fix_end', dest='fix_end', action='store_true', help='fix end point (default: off)')
    parser.set_defaults(init_linear=True)
    parser.add_argument('--init_linear_off', dest='init_linear', action='store_false', help='turns off linear initialization of intermediate points (default: on)')
    parser.add_argument('--resume', type=str, metavar='CKPT', help='checkpoint to resume training from (default: None)')
    parser.add_argument('--epochs', type=int, metavar='N', help='number of epochs to train (default: 200)')
    parser.add_argument('--save_freq', type=int, metavar='N', help='save frequency (default: 50)')
    parser.add_argument('--lr', type=float, metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, metavar='WD', help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', type=int, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--no-compile', action='store_true', help='disable PyTorch 2.0 compilation (default: False)')

    if args is not None:
        # If args is provided, update defaults
        default_args = parser.parse_args([])
        for k, v in vars(args).items():
            setattr(default_args, k, v)
        return default_args
    
    return parser.parse_args()

def run(args):
    """Run the training process with the provided arguments.
    
    Args:
        args: Arguments for the training process. Missing values will be filled with defaults.
    """
    # Define all default values here
    DEFAULTS = {
        'experiment_name': 'default',
        'dir': '/tmp/curve/',
        'loss_config': None,
        'dataset': 'CIFAR10',
        'loss_spec': None,
        'use_test': False,
        'transform': 'VGG',
        'data_path': None,
        'batch_size': 128,
        'num_workers': 4,
        'curve': None,
        'num_bends': 3,
        'init_start': None,
        'fix_start': False,
        'init_end': None,
        'fix_end': False,
        'init_linear': True,
        'resume': None,
        'epochs': 50,
        'save_freq': 5,
        'lr': 0.05,
        'momentum': 0.9,
        'wd': 1e-4,
        'seed': 1,
        'no_compile': False,
    }

    # Fill in missing values with defaults
    for key, value in DEFAULTS.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    # Special case: if model is required and not provided, raise error
    if not hasattr(args, 'model') or args.model is None:
        raise ValueError("The 'model' argument is required but was not provided")
    
    # Set up output directory
    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    # Set PyTorch settings
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test
    )
    print(f"There are {len(loaders)} loaders, which are {list(loaders.keys())}")

    architecture = getattr(models, args.model)

    # Model initialization remains the same as train.py

    if args.curve is None:
        model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    else:
        curve = getattr(curves, args.curve)
        model = curves.CurveNet(
            num_classes,
            curve,
            architecture.curve,
            args.num_bends,
            args.fix_start,
            args.fix_end,
            architecture_kwargs=architecture.kwargs,
        )
        base_model = None
        if args.resume is None:
            for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
                if path is not None:
                    if base_model is None:
                        base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
                    checkpoint = torch.load(path)
                    print('Loading %s as point #%d' % (path, k))
                    base_model.load_state_dict(checkpoint['model_state'])
                    model.import_base_parameters(base_model, k)
            if args.init_linear:
                print('Linear initialization.')
                model.init_linear()
    model.cuda()

    if hasattr(model, 'enable_checkpointing'):
        model.enable_checkpointing()

    if not args.no_compile and not 'vgg' in args.model.lower():
        try:
            # Use more compatible compilation settings
            model = torch.compile(
                model,
                mode='default',
                fullgraph=False,
                dynamic=True  # Enable dynamic shapes
            )
            print("Model compiled with PyTorch 2.0 using default mode with dynamic shapes")
        except Exception as e:
            print(f"PyTorch 2.0 compilation not available: {e}")
    else:
        print("PyTorch 2.0 compilation disabled")

    def learning_rate_schedule(base_lr, epoch, total_epochs):
        alpha = epoch / total_epochs
        if alpha <= 0.5:
            factor = 1.0
        elif alpha <= 0.9:
            factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            factor = 0.01
        return factor * base_lr

    # Load loss configuration
    loss_config = load_loss_config(args.loss_config)

    # Initialize loss tracker, optimizer, and other components
    main_loss = losses.CrossEntropyLoss(name='main_ce')
    auxiliary_losses = {}
    if args.loss_config:
        loss_config = load_loss_config(args.loss_config)
        for i, loss_type in enumerate(loss_config.get('auxiliary_losses', [])):
            aux_loss = losses.CrossEntropyLoss(name=f'aux_ce_{i}')
            print(f"Adding auxiliary Cross Entropy Loss")
            auxiliary_losses[aux_loss.name] = aux_loss

    loss_tracker = losses.LossTracker(main_loss, auxiliary_losses)
    scaler = GradScaler()

    regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd if args.curve is None else 0.0
    )

    # Training loop
    start_epoch = 1
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'tr_f1', 'tr_roc_auc']
    for name in auxiliary_losses:
        columns.append(f'tr_{name}')
    columns.extend(['te_loss', 'te_acc', 'te_f1', 'te_roc_auc'])
    for name in auxiliary_losses:
        columns.append(f'te_{name}')
    columns.append('time')

    utils.save_checkpoint(
        args.dir,
        start_epoch - 1,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )

    has_bn = utils.check_bn(model)
    
    # Set up MLflow experiment
    experiment = args.experiment_name if args.experiment_name else model
    mlflow.set_experiment(experiment)
    if args.experiment_name:
        with mlflow.start_run() as run:
            initial_run_id = run.info.run_id
            mlflow.log_params({
                "experiment_path": args.experiment_path,
                "max_epochs": args.max_epochs,
                "early_stopping_metric": args.early_stopping['metric'],
                "early_stopping_threshold": args.early_stopping['threshold'],
                "test_size": args.test_size,
                "random_seed": args.random_seed
            })
            with open(args.experiment_path, 'r') as f:
                experiment_json = json.load(f)
            mlflow.log_dict(experiment_json, "experiment_config.json")
    print("The run id is:", initial_run_id)
    with mlflow.start_run(run_id=initial_run_id) as run:
        print("The MLFlow verison is:", mlflow.__version__)
        for epoch in range(start_epoch, args.epochs + 1):
            time_ep = time.time()

            lr = learning_rate_schedule(args.lr, epoch, args.epochs)
            utils.adjust_learning_rate(optimizer, lr)

            train_metrics = train_epoch(
                model,
                loaders['train'],
                optimizer,
                scaler,
                loss_tracker,
                regularizer
            )

            test_metrics = test(
                model,
                loaders['test'], #if args.use_test else loaders['valid'],
                loss_tracker,
                regularizer
            )

            if has_bn:
                utils.bn_update(loaders['train'], model)

            time_ep = time.time() - time_ep


            values = [epoch, lr, train_metrics['loss'], train_metrics['accuracy'],
                    train_metrics['f1'],  train_metrics.get('roc_auc', 'N/A')]
            
            for name in auxiliary_losses:
                values.append(train_metrics.get(name, 'N/A'))
            
            values.extend([test_metrics['loss'], test_metrics['accuracy'],
                        test_metrics['f1'], test_metrics.get('roc_auc', 'N/A')])
            
            for name in auxiliary_losses:
                values.append(test_metrics.get(name, 'N/A'))
            
            values.append(time_ep)

            table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
            if epoch % 40 == 1 or epoch == start_epoch:
                print(table.split('\n')[0])
            print(table.split('\n')[2])

            metrics = {'learning_rate': lr}
            metrics['train_loss'] = train_metrics['loss']
            metrics['train_accuracy'] = train_metrics['accuracy']
            metrics['train_f1'] = train_metrics['f1']
            metrics['train_roc_auc'] = train_metrics.get('roc_auc', 0.0)
            metrics['test_loss'] = test_metrics['loss']
            metrics['test_accuracy'] = test_metrics['accuracy']
            metrics['test_f1'] = test_metrics['f1']
            metrics['test_roc_auc'] = test_metrics.get('roc_auc', 0.0)
            for name in auxiliary_losses:
                metrics[f'train_{name}'] = train_metrics.get(name, 0.0)
                metrics[f'test_{name}'] = test_metrics.get(name, 0.0)
            
            mlflow.log_metrics(metrics, step=epoch)

            if epoch % args.save_freq == 0 or epoch == args.epochs:
                print('Saving checkpoint...')
                utils.save_checkpoint(
                    args.dir,
                    epoch,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict()
                )
                model.eval()
                with torch.no_grad():
                    sample_input = next(iter(loaders['test']))[0][:1].cuda()
                    sample_output = model(sample_input)
                signature = infer_signature(sample_input.cpu().numpy(), sample_output.detach().cpu().numpy())
                env = mlflow.pytorch.get_default_conda_env()

                #TODO: change where the model is logged
                experiment_path 
                name = f''
                mlflow.pytorch.log_model(pytorch_model=model, name=f"model-epoch-{epoch}", signature=signature, conda_env=env)


# TODO: add checker and logging of fulfillment of accuracy requirement

def main():
    """Main entry point that parses args and runs the training."""
    args = get_args()
    return run(args)

if __name__ == '__main__':
    main()
