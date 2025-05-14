import argparse
import numpy as np
import os
import json
import torch
from multiprocessing import get_context
import data, models, utils, losses
import time
# Global variables for worker processes
globals_dict = {}

def init_worker(args):
    """
    Initializer for each worker process: load model, data, loss, and plane parameters
    """
    (dataset, data_path, batch_size, num_workers, transform, use_test,
     checkpoint_paths, loss_configs, wd,
     w0, u, v, dx, dy) = args
    
    loaders, num_classes = data.loaders(
        dataset, data_path, batch_size, num_workers, transform, use_test, shuffle_train=False
    )

    regularizer = utils.l2_regularizer(wd)

    model_type = open(checkpoint_paths[0].split('/artifacts')[0] + '/params/model').read().strip()
    arch = getattr(models, model_type)
    model = arch.base(num_classes, **arch.kwargs).cuda()
    model.eval()

    main_loss_fn = losses.get_loss(loss_configs[0]['type'], **loss_configs[0].get('params', {}))
    aux_losses = {}
    for idx, cfg in enumerate(loss_configs[1:], start=1):
        aux_losses[f'model_{idx}_loss'] = losses.get_loss(cfg['type'], **cfg.get('params', {}))
    loss_tracker = losses.LossTracker(main_loss_fn, auxiliary_losses=aux_losses)

    globals_dict['loaders'] = loaders
    globals_dict['model_template'] = model
    globals_dict['regularizer'] = regularizer
    globals_dict['loss_tracker'] = loss_tracker
    globals_dict['w0'] = w0
    globals_dict['u'] = u
    globals_dict['v'] = v
    globals_dict['dx'] = dx
    globals_dict['dy'] = dy


def worker(task):
    print(f"Processing task: {task}")
    start = time.time()
    i, j, alpha, beta = task

    # Clone model template
    model = globals_dict['model_template'].__class__.__new__(globals_dict['model_template'].__class__)
    model.__dict__ = globals_dict['model_template'].__dict__.copy()
    model.cuda()

    # Interpolate weights
    p = globals_dict['w0'] + alpha * globals_dict['dx'] * globals_dict['u'] + beta * globals_dict['dy'] * globals_dict['v']
    offset = 0
    for param in model.parameters():
        num = int(np.prod(param.size()))
        chunk = p[offset:offset+num].reshape(param.size())
        param.data.copy_(torch.from_numpy(chunk))
        offset += num

    utils.update_bn(globals_dict['loaders']['train'], model)

    tr_res = utils.test(globals_dict['loaders']['train'], model, globals_dict['loss_tracker'], globals_dict['regularizer'])
    te_res = utils.test(globals_dict['loaders']['test'],  model, globals_dict['loss_tracker'], globals_dict['regularizer'])
    duration = time.time() - start
    print(f"Processed task ({i}, {j}) in {duration:.2f} seconds")
    return (i, j, tr_res, te_res)


def main():
    main_start = time.time()
    print(f"Main process started at {main_start:.2f} seconds")
    parser = argparse.ArgumentParser(description='Parallel checkpoint plane')
    parser.add_argument('--dir', type=str, default='/tmp/plane', help='output directory')
    parser.add_argument('--grid_points', type=int, default=21)
    parser.add_argument('--margin_left', type=float, default=0.2)
    parser.add_argument('--margin_right', type=float, default=0.2)
    parser.add_argument('--margin_bottom', type=float, default=0.2)
    parser.add_argument('--margin_top', type=float, default=0.2)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--use_test', action='store_true')
    parser.add_argument('--transform', type=str, default='VGG')
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_0', type=str, required=True)
    parser.add_argument('--checkpoint_1', type=str, required=True)
    parser.add_argument('--checkpoint_2', type=str, required=True)
    parser.add_argument('--loss_config', type=str, required=True)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--num_processes', type=int, default=4, help='number of parallel processes')
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    checkpoints = [args.checkpoint_0, args.checkpoint_1, args.checkpoint_2]

    # TODO: use mlruns loss config instead of as argument
    with open(args.loss_config, 'r') as f:
        config = json.load(f)
    loss_cfgs = [config['main_loss']] + config.get('auxiliary_losses', [])

    model_type = open(checkpoints[0].split('/artifacts')[0] + '/params/model').read().strip()
    arch = getattr(models, model_type)
    # TODO: set number of classes based on dataset
    print(f"Model type: {model_type}")
    print(f"Model architecture: {arch}")
    print(f"Checkpoints: {checkpoints}")
    temp_model = arch.base(10, **arch.kwargs).cuda()  # placeholder for weights shape
    w0 = utils.get_weights_from_checkpoint(temp_model, checkpoints[0])
    w1 = utils.get_weights_from_checkpoint(temp_model, checkpoints[1])
    w2 = utils.get_weights_from_checkpoint(temp_model, checkpoints[2])

    u = w1 - w0
    dx = np.linalg.norm(u)
    u = u / dx

    v = w2 - w0
    v = v - np.dot(u, v) * u
    dy = np.linalg.norm(v)
    v = v / dy

    G = args.grid_points
    alphas = np.linspace(-args.margin_left, 1+args.margin_right, G)
    betas = np.linspace(-args.margin_bottom, 1+args.margin_top, G)

    tasks = [(i, j, alpha, beta) for i, alpha in enumerate(alphas) for j, beta in enumerate(betas)]

    init_args = (
        args.dataset, args.data_path, args.batch_size, args.num_workers,
        args.transform, args.use_test, checkpoints, loss_cfgs,
        args.wd, w0, u, v, dx, dy
    )

    ctx = get_context('spawn')
    with ctx.Pool(processes=args.num_processes, initializer=init_worker, initargs=(init_args,)) as pool:
        results = []
        for result in pool.imap_unordered(worker, tasks):
            start = time.time()
            print(f"Worker started task ({result[0]}, {result[1]})")
            results.append(result)
            i, j, _, _ = result
            print(f"Worker completed task ({i}, {j}) in {time.time() - start:.2f} seconds")

    tr_loss = np.zeros((G, G)); te_loss = np.zeros((G, G))
    tr_acc = np.zeros((G, G)); te_acc = np.zeros((G, G))
    for i, j, tr_res, te_res in results:
        tr_loss[i, j] = tr_res['loss']
        tr_acc[i, j]  = tr_res['accuracy']
        te_loss[i, j] = te_res['loss']
        te_acc[i, j]  = te_res['accuracy']

    np.savez(os.path.join(args.dir, 'checkpoint_plane_parallel.npz'),
             alphas=alphas, betas=betas,
             tr_loss=tr_loss, tr_acc=tr_acc,
             te_loss=te_loss, te_acc=te_acc,
             checkpoints=checkpoints)
    print(f"Main process completed in {time.time() - main_start:.2f} seconds")

if __name__ == '__main__':
    main()
