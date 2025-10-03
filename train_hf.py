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


def compute_metrics(eval_pred, loss_tracker):
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


def main():
    parser = argparse.ArgumentParser(description='DNN curve training with HuggingFace metrics')
    parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                        help='training directory (default: /tmp/curve/)')
    parser.add_argument('--loss_config', type=str, default='loss_config.json',
                        help='path to loss configuration file (default: loss_config.json)')
    
    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--use_test', default=True, action='store_true',
                        help='switches between validation and test set (default: test)')
    parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                        help='transform name (default: VGG)')
    parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                        help='path to datasets location (default: None)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')

    parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                        help='model name (default: None)')

    parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                        help='curve type to use (default: None)')
    parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                        help='number of curve bends (default: 3)')
    parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                        help='checkpoint to init start point (default: None)')
    parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                        help='fix start point (default: off)')
    parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                        help='checkpoint to init end point (default: None)')
    parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                        help='fix end point (default: off)')
    parser.set_defaults(init_linear=True)
    parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                        help='turns off linear initialization of intermediate points (default: on)')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                        help='save frequency (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--no-compile', action='store_true',
                        help='disable PyTorch 2.0 compilation (default: False)')

    args = parser.parse_args()
    
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
    # ... [Previous model initialization code] ...

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
                train_metrics['f1'], train_metrics.get('roc_auc', 'N/A')]
        
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

        utils.save_checkpoint(
            args.dir,
            epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

        # Log metrics to MLflow
        mlflow.set_experiment(args.model)
        with mlflow.start_run():
            mlflow.log_param("epoch", epoch)
            mlflow.log_param("learning_rate", lr)
            
            # Log training metrics
            mlflow.log_metrics({
                "train_loss": train_metrics['loss'],
                "train_accuracy": train_metrics['accuracy'],
                "train_f1": train_metrics['f1'],
                "train_roc_auc": train_metrics.get('roc_auc', 0.0)
            })
            
            # Log test metrics
            mlflow.log_metrics({
                "test_loss": test_metrics['loss'],
                "test_accuracy": test_metrics['accuracy'],
                "test_f1": test_metrics['f1'],
                "test_roc_auc": test_metrics.get('roc_auc', 0.0)
            })


if __name__ == '__main__':
    main()
