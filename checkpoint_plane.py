import argparse
import numpy as np
import os
import json
import tabulate
import torch
import torch.nn.functional as F
import time
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
parser.add_argument('--checkpoint_0', type=str, required=True,
                    help='first checkpoint file to use as reference point')
parser.add_argument('--checkpoint_1', type=str, required=True,
                    help='second checkpoint file to use as reference point')
parser.add_argument('--checkpoint_2', type=str, required=True,
                    help='third checkpoint file to use as reference point')
parser.add_argument('--loss_config', type=str, required=True,
                    help='path to loss configuration JSON file')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

def load_loss_config(config_path):
    """Load loss configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_xy(point, origin, vector_x, vector_y):
    """Get 2D coordinates of a point in the plane"""
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

def main():
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    checkpoints = [args.checkpoint_0, args.checkpoint_1, args.checkpoint_2]

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
    

    regularizer = utils.l2_regularizer(args.wd)

    # Get weights from all checkpoints
    # The model architecture needs to be the same for all checkpoints because otherwise you can't interpolate between them
    with open(args.checkpoint_0.split('/artifacts')[0] + '/params/model', 'r') as f:
        model_type = f.read()
    model_architecture = getattr(models, model_type)
    base_model = model_architecture.base(num_classes, **model_architecture.kwargs)
    base_model.cuda()


    loss_config_path = os.listdir(args.checkpoint_0.split('/model')[0] + '/loss_config.json')[0]
    with open(loss_config_path, 'r') as f:
        model_0_loss_config = json.load(f)['main_loss']

    # Load loss configuration
    model_0_criterion = losses.get_loss(model_0_loss_config['type'], **model_0_loss_config.get('params', {}))

    w_0 = utils.get_weights_from_checkpoint(base_model, args.checkpoint_0)

    loss_config_path = os.listdir(args.checkpoint_1.split('/model')[0] + '/loss_config.json')[0]
    with open(loss_config_path, 'r') as f:
        model_1_loss_config = json.load(f)['main_loss']

    # Load loss configuration
    model_1_criterion = losses.get_loss(model_1_loss_config['type'], **model_1_loss_config.get('params', {}))

    w_1 = utils.get_weights_from_checkpoint(base_model, args.checkpoint_1)

    loss_config_path = os.listdir(args.checkpoint_2.split('/model')[0] + '/loss_config.json')[0]
    with open(loss_config_path, 'r') as f:
        model_2_loss_config = json.load(f)['main_loss']

    # Load loss configuration
    model_2_criterion = losses.get_loss(model_2_loss_config['type'], **model_2_loss_config.get('params', {}))
    

    w_2 = utils.get_weights_from_checkpoint(base_model, args.checkpoint_2)

    loss_tracker = losses.LossTracker(model_0_criterion, auxiliary_losses={
        'model_1_loss': model_1_criterion,
        'model_2_loss': model_2_criterion})


    # Create plane from first three points
    u = w_1 - w_0  # First basis vector
    dx = np.linalg.norm(u)
    u /= dx

    v = w_2 - w_0  # Second basis vector
    v -= np.dot(u, v) * u  # Make orthogonal to u
    dy = np.linalg.norm(v)
    v /= dy

    # Store all reference point coordinates
    w_0_ref_coordinates = get_xy(w_0, w_0, u, v) 
    w_1_ref_coordinates = get_xy(w_1, w_0, u, v)
    w_2_ref_coordinates = get_xy(w_2, w_0, u, v)

    ref_coordinates = np.stack([w_0_ref_coordinates, w_1_ref_coordinates, w_2_ref_coordinates] )


    # Setup evaluation grid
    G = args.grid_points
    alphas = np.linspace(0.0 - args.margin_left, 1.0 + args.margin_right, G)
    betas = np.linspace(0.0 - args.margin_bottom, 1.0 + args.margin_top, G)

    # Initialize result arrays
    tr_loss = np.zeros((G, G))
    tr_loss_1 = np.zeros((G, G))
    tr_loss_2 = np.zeros((G, G))
    tr_nll = np.zeros((G, G))
    tr_acc = np.zeros((G, G))
    tr_err = np.zeros((G, G))
    te_loss = np.zeros((G, G))
    te_loss_1 = np.zeros((G, G))
    te_loss_2 = np.zeros((G, G))
    te_nll = np.zeros((G, G))
    te_acc = np.zeros((G, G))
    te_err = np.zeros((G, G))
    grid = np.zeros((G, G, 2))

    # columns = ['X', 'Y', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']
    start = time.time()
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Get point in the plane
            p = w_0 + alpha * dx * u + beta * dy * v

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
            # TODO: this should compute *all* relevant losses, not just the main one

            tr_res = utils.test(loaders['train'], base_model, loss_tracker, regularizer)
            te_res = utils.test(loaders['test'], base_model, loss_tracker, regularizer)

            # Store results
            tr_loss[i, j] = tr_res['loss']
            tr_loss_1[i, j] = tr_res['model_1_loss']
            tr_loss_2[i, j] = tr_res['model_2_loss']
            # tr_nll[i, j] = tr_res['nll']
            tr_acc[i, j] = tr_res['accuracy']
            tr_err[i, j] = 100.0 - tr_acc[i, j]

            te_loss[i, j] = te_res['loss']
            te_loss_1[i, j] = te_res['model_1_loss']
            te_loss_2[i, j] = te_res['model_2_loss']
            # te_nll[i, j] = te_res['nll']
            te_acc[i, j] = te_res['accuracy']
            te_err[i, j] = 100.0 - te_acc[i, j]

            grid[i, j] = [alpha * dx, beta * dy]

            # # Print current results
            # values = [
            #     grid[i, j, 0], grid[i, j, 1], tr_loss[i, j], tr_nll[i, j], tr_err[i, j],
            #     te_nll[i, j], te_err[i, j]
            # ]
            # table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
            # if j == 0:
            #     table = table.split('\n')
            #     table = '\n'.join([table[1]] + table)
            # else:
            #     table = table.split('\n')[2]
            # print(table)
            if (i * G + j + 1) % 10 == 0:
                # Print progress every 10 points
                print(f"Out of all {G*G} points, {i*G+j+1} done ({(i*G+j+1)/(G*G)*100:.2f}%) in {time.time() - start:.2f} seconds")
                start = time.time()

    # Save results
    np.savez(
        os.path.join(args.dir, 'checkpoint_plane.npz'),
        ref_coordinates=ref_coordinates,
        alphas=alphas,
        betas=betas,
        grid=grid,
        tr_loss=tr_loss,
        tr_loss_1=tr_loss_1,
        tr_loss_2=tr_loss_2,
        tr_acc=tr_acc,
        tr_nll=tr_nll,
        tr_err=tr_err,
        te_loss=te_loss,
        te_loss_1=te_loss_1,
        te_loss_2=te_loss_2,
        te_acc=te_acc,
        te_nll=te_nll,
        te_err=te_err,
        checkpoints=checkpoints,
    )

if __name__ == '__main__':
    main()