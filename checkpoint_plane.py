import argparse
import numpy as np
import os
import json
import tabulate
import torch
import torch.nn.functional as F

import data
import models
import utils
import losses

parser = argparse.ArgumentParser(description='Computes values for plane visualization using arbitrary checkpoints')
parser.add_argument('--dir', type=str, default='/tmp/plane', metavar='DIR',
                    help='training directory (default: /tmp/plane)')
parser.add_argument('--grid_points', type=int, default=21, metavar='N',
                    help='number of points in the grid (default: 21)')
parser.add_argument('--margin_left', type=float, default=0.2, metavar='M',
                    help='left margin (default: 0.2)')
parser.add_argument('--margin_right', type=float, default=0.2, metavar='M',
                    help='right margin (default: 0.2)')
parser.add_argument('--margin_bottom', type=float, default=0.2, metavar='M',
                    help='bottom margin (default: 0.)')
parser.add_argument('--margin_top', type=float, default=0.2, metavar='M',
                    help='top margin (default: 0.2)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default='data', metavar='PATH',
                    help='path to datasets location (default: data)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--checkpoints', nargs='+', type=str, required=True,
                    help='list of checkpoint files to use as reference points')
parser.add_argument('--loss_config', type=str, required=True,
                    help='path to loss configuration JSON file')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

def load_loss_config(config_path):
    """Load loss configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_weights_from_checkpoint(model, checkpoint_path):
    """Extract weights from a checkpoint file"""
    checkpoint = torch.load(checkpoint_path)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return np.concatenate([p.data.cpu().numpy().ravel() for p in model.parameters()])

def get_xy(point, origin, vector_x, vector_y):
    """Get 2D coordinates of a point in the plane"""
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

def main():
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    if len(args.checkpoints) < 3:
        raise ValueError("At least 3 checkpoints are required to define a plane")

    # Setup data loaders
    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test,
        shuffle_train=False
    )

    # Setup model
    architecture = getattr(models, args.model)
    base_model = architecture.base(num_classes, **architecture.kwargs)
    base_model.cuda()

    # Load loss configuration
    loss_config = load_loss_config(args.loss_config)
    main_loss_config = loss_config.get('main_loss', {'type': 'ce'})
    print('Using main loss:', main_loss_config['type'], 'with params:', main_loss_config.get('params', {}))
    criterion = losses.get_loss(main_loss_config['type'], **main_loss_config.get('params', {}))
    loss_tracker = losses.LossTracker(criterion)

    regularizer = utils.l2_regularizer(args.wd)

    # Get weights from all checkpoints
    w = [get_weights_from_checkpoint(base_model, ckpt) for ckpt in args.checkpoints]
    print('Weight space dimensionality: %d' % w[0].shape[0])

    # Create plane from first three points
    u = w[1] - w[0]  # First basis vector
    dx = np.linalg.norm(u)
    u /= dx

    v = w[2] - w[0]  # Second basis vector
    v -= np.dot(u, v) * u  # Make orthogonal to u
    dy = np.linalg.norm(v)
    v /= dy

    # Store all reference point coordinates
    ref_coordinates = np.stack([get_xy(p, w[0], u, v) for p in w])

    # Setup evaluation grid
    G = args.grid_points
    alphas = np.linspace(0.0 - args.margin_left, 1.0 + args.margin_right, G)
    betas = np.linspace(0.0 - args.margin_bottom, 1.0 + args.margin_top, G)

    # Initialize result arrays
    tr_loss = np.zeros((G, G))
    tr_nll = np.zeros((G, G))
    tr_acc = np.zeros((G, G))
    tr_err = np.zeros((G, G))
    te_loss = np.zeros((G, G))
    te_nll = np.zeros((G, G))
    te_acc = np.zeros((G, G))
    te_err = np.zeros((G, G))
    grid = np.zeros((G, G, 2))

    columns = ['X', 'Y', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Get point in the plane
            p = w[0] + alpha * dx * u + beta * dy * v

            # Update model parameters
            offset = 0
            for parameter in base_model.parameters():
                size = np.prod(parameter.size())
                value = p[offset:offset+size].reshape(parameter.size())
                parameter.data.copy_(torch.from_numpy(value))
                offset += size

            # Update batch normalization statistics
            utils.update_bn(loaders['train'], base_model)

            # Evaluate model
            tr_res = utils.test(loaders['train'], base_model, loss_tracker, regularizer)
            te_res = utils.test(loaders['test'], base_model, loss_tracker, regularizer)

            # Store results
            tr_loss[i, j] = tr_res['loss']
            # tr_nll[i, j] = tr_res['nll']
            tr_acc[i, j] = tr_res['accuracy']
            tr_err[i, j] = 100.0 - tr_acc[i, j]

            te_loss[i, j] = te_res['loss']
            # te_nll[i, j] = te_res['nll']
            te_acc[i, j] = te_res['accuracy']
            te_err[i, j] = 100.0 - te_acc[i, j]

            grid[i, j] = [alpha * dx, beta * dy]

            # Print current results
            values = [
                grid[i, j, 0], grid[i, j, 1], tr_loss[i, j], tr_nll[i, j], tr_err[i, j],
                te_nll[i, j], te_err[i, j]
            ]
            table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
            if j == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table)

    # Save results
    np.savez(
        os.path.join(args.dir, 'checkpoint_plane.npz'),
        ref_coordinates=ref_coordinates,
        alphas=alphas,
        betas=betas,
        grid=grid,
        tr_loss=tr_loss,
        tr_acc=tr_acc,
        tr_nll=tr_nll,
        tr_err=tr_err,
        te_loss=te_loss,
        te_acc=te_acc,
        te_nll=te_nll,
        te_err=te_err
    )

if __name__ == '__main__':
    main()