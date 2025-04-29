import argparse
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch as ch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim import lr_scheduler
import torchvision.transforms as T
import tabulate

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

import curves
import models
import utils

def main():
    parser = argparse.ArgumentParser(description='DNN curve training with FFCV')
    parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                        help='training directory (default: /tmp/curve/)')
    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--use_test', action='store_true',
                        help='switches between validation and test set (default: validation)')
    parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                        help='path to datasets location (default: None)')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size (default: 512)')
    parser.add_argument('--num-workers', type=int, default=8, metavar='N',
                        help='number of workers (default: 8)')
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
    parser.add_argument('--wd', type=float, default=5e-4, metavar='WD',
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ffcv_dataset', type=str, required=True,
                        help='path to FFCV dataset file')
    parser.add_argument('--ffcv_val_dataset', type=str, required=True,
                        help='path to FFCV validation dataset file')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='label smoothing value (default: 0.1)')
    parser.add_argument('--lr_peak_epoch', type=int, default=2, metavar='N',
                        help='epoch at which learning rate peaks (default: 2)')
    parser.add_argument('--lr_tta', action='store_true',
                        help='use test time augmentation (default: False)')

    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    ch.backends.cudnn.benchmark = True
    ch.manual_seed(args.seed)
    ch.cuda.manual_seed(args.seed)

    # CIFAR normalization values (scaled for float32)
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]

    # FFCV data loading pipeline
    device = ch.device('cuda:0')
    image_pipeline = [
        SimpleRGBImageDecoder(),
        RandomHorizontalFlip(),
        RandomTranslate(padding=2, fill=tuple(map(int, [125.307, 122.961, 113.8575]))),
        Cutout(4, tuple(map(int, [125.307, 122.961, 113.8575]))),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        Convert(ch.float32),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        Squeeze(),
    ]

    # Create FFCV loaders
    train_loader = Loader(
        args.ffcv_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=OrderOption.RANDOM,
        drop_last=True,
        pipelines={
            'image': image_pipeline,
            'label': label_pipeline
        }
    )

    val_loader = Loader(
        args.ffcv_val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=OrderOption.SEQUENTIAL,
        pipelines={
            'image': [SimpleRGBImageDecoder(), ToTensor(), ToDevice(device, non_blocking=True), 
                     ToTorchImage(), Convert(ch.float32), T.Normalize(CIFAR_MEAN, CIFAR_STD)],
            'label': label_pipeline
        }
    )

    loaders = {
        'train': train_loader,
        'test': val_loader
    }
    num_classes = 10 if args.dataset == 'CIFAR10' else 100

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
                    checkpoint = ch.load(path)
                    print('Loading %s as point #%d' % (path, k))
                    base_model.load_state_dict(checkpoint['model_state'])
                    model.import_base_parameters(base_model, k)
            if args.init_linear:
                print('Linear initialization.')
                model.init_linear()
    model = model.to(memory_format=ch.channels_last).cuda()

    # Setup optimizer and learning rate schedule
    criterion = F.cross_entropy
    regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)
    optimizer = SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd if args.curve is None else 0.0
    )

    # Cyclic learning rate schedule
    iters_per_epoch = len(loaders['train'])
    lr_schedule = np.interp(
        np.arange((args.epochs + 1) * iters_per_epoch),
        [0, args.lr_peak_epoch * iters_per_epoch, args.epochs * iters_per_epoch],
        [0, 1, 0]
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    start_epoch = 1
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = ch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'data_time', 'train_time', 'eval_time', 'total_time']

    utils.save_checkpoint(
        args.dir,
        start_epoch - 1,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )

    has_bn = utils.check_bn(model)
    test_res = {'loss': None, 'accuracy': None, 'nll': None}
    
    def train_epoch(loader, model, criterion, optimizer, scheduler):
        total_time = time.time()
        data_time = 0
        train_time = 0
        
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(loader):
            data_start = time.time()
            data_time += data_start - total_time
            
            train_start = time.time()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_time += time.time() - train_start
            total_time = time.time()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}: Data time: {data_time:.3f}s, Train time: {train_time:.3f}s')
        
        return total_loss / total_samples, total_correct / total_samples, data_time, train_time

    def test_epoch(loader, model, criterion):
        eval_start = time.time()
        model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with ch.no_grad():
            for inputs, targets in loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += inputs.size(0)
        
        eval_time = time.time() - eval_start
        return total_loss / total_samples, total_correct / total_samples, eval_time

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        # Learning rate schedule
        if epoch <= args.lr_peak_epoch:
            lr = args.lr * (epoch / args.lr_peak_epoch)
        else:
            lr = args.lr * (1 - (epoch - args.lr_peak_epoch) / (args.epochs - args.lr_peak_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Training
        train_loss, train_acc, data_time, train_time = train_epoch(loaders['train'], model, criterion, optimizer, scheduler)
        
        # Evaluation
        test_loss, test_acc, eval_time = test_epoch(loaders['test'], model, criterion)
        
        total_time = time.time() - epoch_start
        
        values = [epoch, lr, train_loss, train_acc * 100, test_loss, test_acc * 100,
                 data_time, train_time, eval_time, total_time]
        
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 1:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

        if epoch % args.save_freq == 0:
            utils.save_checkpoint(
                args.dir,
                epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

    if args.epochs % args.save_freq != 0:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

if __name__ == '__main__':
    main() 