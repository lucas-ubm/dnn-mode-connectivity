import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import seaborn as sns
import mlflow
import json

parser = argparse.ArgumentParser(description='Checkpoint plane visualization')
parser.add_argument('--dir', type=str, default='/tmp/checkpoint_plane/', metavar='DIR',
                    help='directory containing checkpoint plane data (default: /tmp/checkpoint_plane/)')

class LogNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        log_v = np.ma.log(value - self.vmin)
        log_v = np.ma.maximum(log_v, self.log_alpha)
        return 0.9 * (log_v - self.log_alpha) / (np.log(self.vmax - self.vmin) - self.log_alpha)

def plot_plane(file, args, ref_coordinates, property: str):
    plt.figure(figsize=(12.4, 7))

    contour, contourf, colorbar = plane(
        file['grid'],
        file[property],
        vmax=5.0,
        log_alpha=-5.0,
        N=7
    )
    mlflow_runs = load_mlflow_runs(file)

    ref_coordinates = file['ref_coordinates']
    print(mlflow_runs[0])
    plt.scatter(ref_coordinates[1, 0], ref_coordinates[1, 1], marker='o', c='k', s=120, zorder=2, label=mlflow_runs[1].info.run_name)
    plt.scatter(ref_coordinates[2, 0], ref_coordinates[2, 1], marker='D', c='k', s=120, zorder=2, label=mlflow_runs[2].info.run_name)
    plt.scatter(ref_coordinates[0, 0], ref_coordinates[0, 1], marker='^', c='k', s=120, zorder=2, label=mlflow_runs[0].info.run_name)

    plt.legend(fontsize=14)
    plt.margins(0.0)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    colorbar.ax.tick_params(labelsize=18)
    plt.savefig(os.path.join(args.dir, f'{property}_plane.pdf'), format='pdf', bbox_inches='tight')

def plane(grid, values, vmax=None, log_alpha=-5, N=7, cmap='jet_r'):
    cmap = plt.get_cmap(cmap)
    if vmax is None:
        clipped = values.copy()
    else:
        clipped = np.minimum(values, vmax)
    log_gamma = (np.log(clipped.max() - clipped.min()) - log_alpha) / N
    levels = clipped.min() + np.exp(log_alpha + log_gamma * np.arange(N + 1))
    levels[0] = clipped.min()
    levels[-1] = clipped.max()
    levels = np.concatenate((levels, [1e10]))
    norm = LogNormalize(clipped.min() - 1e-8, clipped.max() + 1e-8, log_alpha=log_alpha)
    contour = plt.contour(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, norm=norm,
                         linewidths=2.5,
                         zorder=1,
                         levels=levels)
    contourf = plt.contourf(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, norm=norm,
                           levels=levels,
                           zorder=0,
                           alpha=0.55)
    colorbar = plt.colorbar(format='%.2g')
    labels = list(colorbar.ax.get_yticklabels())
    labels[-1].set_text(r'$>\,$' + labels[-2].get_text())
    colorbar.ax.set_yticklabels(labels)
    return contour, contourf, colorbar

def load_mlflow_runs(file):
    checkpoints = file['checkpoints']
    runs = []
    # TODO: take out .item().values() when I save the checkpoints as array instead of dict
    for checkpoint in checkpoints.item().values():
        run_id = checkpoint.split('/artifacts')[0].split('/')[-1]  # Assuming checkpoint names are stored as bytes
        try:
            run = mlflow.get_run(run_id)
            runs.append(run)
        except mlflow.exceptions.RestException:
            print(f"Run with ID {run_id} not found in MLflow tracking server.")
    return runs

def main():
    args = parser.parse_args()

    # TODO: switch back to allow_pickle=False when I save the checkpoints as array instead of dict
    file = np.load(os.path.join(args.dir, 'checkpoint_plane.npz'), allow_pickle=True)

    matplotlib.rc('xtick.major', pad=12)
    matplotlib.rc('ytick.major', pad=12)
    matplotlib.rc('grid', linewidth=0.8)

    sns.set_style('whitegrid')

    mlflow_runs = load_mlflow_runs(file)
    print(mlflow_runs)

    # Plot training loss
    plot_plane(file, args, file['ref_coordinates'], 'tr_loss')
    plot_plane(file, args, file['ref_coordinates'], 'tr_loss_1')
    plot_plane(file, args, file['ref_coordinates'], 'tr_loss_2')

    # Plot test error
    plot_plane(file, args, file['ref_coordinates'], 'te_err')

if __name__ == "__main__":
    main()
