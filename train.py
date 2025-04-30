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


def main():
    parser = argparse.ArgumentParser(description='DNN curve training')
    parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                        help='training directory (default: /tmp/curve/)')
    parser.add_argument('--loss_config', type=str, default='loss_config.json',
                        help='path to loss configuration file (default: loss_config.json)')

    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--use_test', action='store_true',
                        help='switches between validation and test set (default: validation)')
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

    # Set up MLflow experiment
    mlflow.set_experiment(args.model)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log all hyperparameters
        mlflow.log_params(vars(args))
        # Log loss config as artifact
        mlflow.log_artifact(args.loss_config, 'loss_config.json')


        os.makedirs(args.dir, exist_ok=True)
        with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
            f.write(' '.join(sys.argv))
            f.write('\n')

        # Set PyTorch 2.0 optimizations
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

        architecture = getattr(models, args.model)

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

        if not args.no_compile:
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

        # Initialize main loss function
        if loss_config['main_loss']['type'] == 'focal':
            main_loss = losses.FocalLoss(
                alpha=loss_config['main_loss']['params']['alpha'],
                gamma=loss_config['main_loss']['params']['gamma'],
                name='main_focal'
            )
            print(f"Using Focal Loss with alpha={loss_config['main_loss']['params']['alpha']}, gamma={loss_config['main_loss']['params']['gamma']}")
        else:
            main_loss = losses.CrossEntropyLoss(name='main_ce')
            print("Using Cross Entropy Loss")

        # Initialize auxiliary losses
        auxiliary_losses = {}
        for i, aux_config in enumerate(loss_config['auxiliary_losses']):
            if aux_config['type'] == 'focal':
                aux_loss = losses.FocalLoss(
                    alpha=aux_config['params']['alpha'],
                    gamma=aux_config['params']['gamma'],
                    name=f'aux_focal_{i}'
                )
                print(f"Adding auxiliary Focal Loss with alpha={aux_config['params']['alpha']}, gamma={aux_config['params']['gamma']}")
            else:
                aux_loss = losses.CrossEntropyLoss(name=f'aux_ce_{i}')
                print(f"Adding auxiliary Cross Entropy Loss")
            auxiliary_losses[aux_loss.name] = aux_loss

        # Create loss tracker
        loss_tracker = losses.LossTracker(main_loss, auxiliary_losses)

        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler()

        regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)
        optimizer = torch.optim.SGD(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd if args.curve is None else 0.0
        )

        start_epoch = 1
        if args.resume is not None:
            print('Resume training from %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        # Update columns to include all metrics
        columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'tr_f1', 'tr_roc_auc']
        for name in auxiliary_losses:
            columns.append(f'tr_{name}')
        columns.extend(['te_nll', 'te_acc', 'te_f1', 'te_roc_auc'])
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
        test_res = {'loss': None, 'accuracy': None, 'nll': None, 'f1': None, 'roc_auc': None}
        
        for epoch in range(start_epoch, args.epochs + 1):
            time_ep = time.time()

            lr = learning_rate_schedule(args.lr, epoch, args.epochs)
            utils.adjust_learning_rate(optimizer, lr)

            train_res = utils.train(loaders['train'], model, optimizer, loss_tracker, regularizer, scaler)
            if args.curve is None or not has_bn:
                test_res = utils.test(loaders['test'], model, loss_tracker, regularizer, scaler)

            if epoch % args.save_freq == 0:
                checkpoint_path = os.path.join(args.dir, f'checkpoint-{epoch}.pt')
                utils.save_checkpoint(
                    args.dir,
                    epoch,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict()
                )
                # Log model state dict to MLflow instead of full model
                mlflow.pytorch.log_state_dict(model.state_dict(), f"model-epoch-{epoch}")

            time_ep = time.time() - time_ep
            values = [
                epoch, lr,
                train_res['loss'], train_res['accuracy'], train_res['f1'], train_res['roc_auc']
            ]
            values.extend(train_res.get(name, 0.0) for name in auxiliary_losses)
            values.extend([
                test_res['nll'], test_res['accuracy'], test_res['f1'], test_res['roc_auc']
            ])
            values.extend(test_res.get(name, 0.0) for name in auxiliary_losses)
            values.append(time_ep)

            # Log metrics to MLflow
            metrics_dict = {
                'learning_rate': lr,
                'train_loss': train_res['loss'],
                'train_accuracy': train_res['accuracy'],
                'train_f1': train_res['f1'],
                'train_roc_auc': train_res['roc_auc'],
                'test_nll': test_res['nll'],
                'test_accuracy': test_res['accuracy'],
                'test_f1': test_res['f1'],
                'test_roc_auc': test_res['roc_auc'],
                'time_per_epoch': time_ep
            }
            
            # Add auxiliary losses to metrics
            for name, value in train_res.items():
                if name.startswith('aux_'):
                    metrics_dict[f'train_{name}'] = value
            for name, value in test_res.items():
                if name.startswith('aux_'):
                    metrics_dict[f'test_{name}'] = value
                    
            mlflow.log_metrics(metrics_dict, step=epoch)

            table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
            if epoch % 40 == 1 or epoch == start_epoch:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table)

        if args.epochs % args.save_freq != 0:
            utils.save_checkpoint(
                args.dir,
                args.epochs,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )
            # Log final model state dict to MLflow instead of full model
            mlflow.pytorch.log_state_dict(model.state_dict(), "final_model")


if __name__ == '__main__':
    main()
