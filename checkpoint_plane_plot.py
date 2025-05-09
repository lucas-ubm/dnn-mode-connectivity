import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import seaborn as sns

parser = argparse.ArgumentParser(description='Checkpoint plane visualization')
parser.add_argument('--dir', type=str, default='/tmp/checkpoint_plane/', metavar='DIR',
                    help='directory containing checkpoint plane data (default: /tmp/checkpoint_plane/)')

args = parser.parse_args()

file = np.load(os.path.join(args.dir, 'checkpoint_plane.npz'))

matplotlib.rc('xtick.major', pad=12)
matplotlib.rc('ytick.major', pad=12)
matplotlib.rc('grid', linewidth=0.8)

sns.set_style('whitegrid')

class LogNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        log_v = np.ma.log(value - self.vmin)
        log_v = np.ma.maximum(log_v, self.log_alpha)
        return 0.9 * (log_v - self.log_alpha) / (np.log(self.vmax - self.vmin) - self.log_alpha)

def plot_plane(file, args, ref_coordinates, property:str):
    
    # Plot training loss
    plt.figure(figsize=(12.4, 7))

    contour, contourf, colorbar = plane(
        file['grid'],
        file[property],
        vmax=5.0,
        log_alpha=-5.0,
        N=7
    )

    ref_coordinates = file['ref_coordinates']
    # Plot reference points and connecting line
    plt.scatter(ref_coordinates[1, 0], ref_coordinates[1, 1], marker='o', c='k', s=120, zorder=2, label='Binary CE')
    plt.scatter(ref_coordinates[2, 0], ref_coordinates[2, 1], marker='D', c='k', s=120, zorder=2, label='Focal 1')
    plt.scatter(ref_coordinates[0, 0], ref_coordinates[0, 1], marker='D', c='k', s=120, zorder=2, label='Focal 2')


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

# Plot training loss using function plot_plane 
plot_plane(file, args, file['ref_coordinates'], 'tr_loss')

# Plot test error using function plot_plane
plot_plane(file, args, file['ref_coordinates'], 'te_err')
