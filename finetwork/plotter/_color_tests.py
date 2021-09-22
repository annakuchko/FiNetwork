from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, np.pi, 0.1)
y = np.arange(0, 2 * np.pi, 0.1)
X, Y = np.meshgrid(x, y)
Z = np.cos(X) * np.sin(Y) * 10


colors = [
        '#FEA3A3', '#FFAD78', '#FFCF78', '#FFED78', '#F5FF78', '#D9FF78',
        '#B8FF78', '#78FF7E', '#78FFB2', '#78FFDD', '#78F1FF', '#97C0FF',
        '#B497FF', '#E797FF', '#FCA0F2', '#FCA0CF', '#A0FCEA', '#FFEECC'
        ]
n_bins = [20, 25, 50, 100]  # Discretizes the interpolation into bins
cmap_name = 'my_list'
fig, axs = plt.subplots(2, 2, figsize=(6, 9))
fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
for n_bin, ax in zip(n_bins, axs.ravel()):
    # Create the colormap
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)
    # Fewer bins will result in "coarser" colomap interpolation
    im = ax.imshow(Z, origin='lower', cmap=cmap)
    ax.set_title("N bins: %s" % n_bin)
    fig.colorbar(im, ax=ax)