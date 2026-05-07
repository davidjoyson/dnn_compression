import matplotlib.pyplot as plt
import numpy as np
from .save_utils import fig_path

def plot_scaling(results, filename="scaling_heatmap.png"):
    neurons = sorted(list(set(r[0] for r in results)))
    branches = sorted(list(set(r[1] for r in results)))

    acc_un = np.zeros((len(neurons), len(branches)))
    acc_comp = np.zeros((len(neurons), len(branches)))

    for n, b, au, ac in results:
        i = neurons.index(n)
        j = branches.index(b)
        acc_un[i, j] = au
        acc_comp[i, j] = ac

    fig, ax = plt.subplots(1, 2, figsize=(12,5))

    im1 = ax[0].imshow(acc_un, cmap="viridis")
    ax[0].set_title("Uncompressed Accuracy")
    ax[0].set_xticks(range(len(branches)))
    ax[0].set_yticks(range(len(neurons)))
    ax[0].set_xticklabels(branches)
    ax[0].set_yticklabels(neurons)
    fig.colorbar(im1, ax=ax[0])

    im2 = ax[1].imshow(acc_comp, cmap="viridis")
    ax[1].set_title("Compressed Accuracy")
    ax[1].set_xticks(range(len(branches)))
    ax[1].set_yticks(range(len(neurons)))
    ax[1].set_xticklabels(branches)
    ax[1].set_yticklabels(neurons)
    fig.colorbar(im2, ax=ax[1])

    plt.savefig(fig_path(filename), dpi=300, bbox_inches="tight")
    plt.close()
