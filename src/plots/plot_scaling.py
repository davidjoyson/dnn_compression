import matplotlib.pyplot as plt
from .save_utils import fig_path


def plot_scaling(results, neurons1_list, neurons2_list, branches_list,
                 filename_prefix="scaling"):
    """
    Generates one figure per branch count, each with three subplots:
      - Uncompressed accuracy heatmap
      - Compressed accuracy heatmap
      - Compression ratio heatmap (size_u / size_c)

    results: dict with keys accuracy_uncompressed, accuracy_compressed,
             size_uncompressed, size_compressed — all 3D tensors
             [len(neurons1_list), len(neurons2_list), len(branches_list)]
    """
    acc_u  = results["accuracy_uncompressed"].numpy()
    acc_c  = results["accuracy_compressed"].numpy()
    size_u = results["size_uncompressed"].numpy()
    size_c = results["size_compressed"].numpy()
    ratio  = size_u / size_c.clip(min=1)
    t_sec  = results["time_per_config"].numpy()

    for k, br in enumerate(branches_list):
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle(f"Scaling Experiment — branches={br}", fontsize=13)

        data = [
            (acc_u[:, :, k],  "Uncompressed Accuracy", "viridis", (0, 1)),
            (acc_c[:, :, k],  "Compressed Accuracy",   "viridis", (0, 1)),
            (ratio[:, :, k],  "Compression Ratio",     "plasma",  None),
            (t_sec[:, :, k],  "Time (sec)",            "magma",   None),
        ]

        for ax, (mat, title, cmap, vrange) in zip(axes, data):
            kwargs = {"cmap": cmap, "origin": "lower", "aspect": "auto"}
            if vrange:
                kwargs["vmin"], kwargs["vmax"] = vrange
            im = ax.imshow(mat, **kwargs)
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(neurons2_list)))
            ax.set_xticklabels(neurons2_list)
            ax.set_yticks(range(len(neurons1_list)))
            ax.set_yticklabels(neurons1_list)
            ax.set_xlabel("hidden_neurons2")
            ax.set_ylabel("hidden_neurons1")
            ax.set_title(title)

            for i in range(len(neurons1_list)):
                for j in range(len(neurons2_list)):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if mat[i, j] < 0.6 else "black")

        plt.tight_layout()
        plt.savefig(fig_path(f"{filename_prefix}_branches{br}.png"), dpi=150)
        plt.close()
