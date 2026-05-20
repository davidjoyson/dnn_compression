import numpy as np
import matplotlib.pyplot as plt
from .save_utils import fig_path
from .style import apply_style


def plot_confusion_matrix(conf_matrix_data, title="", filename=None, class_names=None):
    """
    conf_matrix_data: dict with keys "uncompressed" and optionally "compressed" (numpy arrays)
    Plots side-by-side normalized confusion matrices.
    """
    apply_style()

    matrices = {k: v for k, v in conf_matrix_data.items() if v is not None}
    if not matrices:
        return

    n_plots = len(matrices)
    subtitles = {"uncompressed": "Uncompressed", "compressed": "Snowflake (int8)"}

    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.2))
    if n_plots == 1:
        axes = [axes]

    for ax, (key, cm) in zip(axes, matrices.items()):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        n = cm.shape[0]
        labels = class_names if class_names and len(class_names) == n else [str(i) for i in range(n)]

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(subtitles.get(key, key), fontsize=10, fontweight="bold", pad=8)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        thresh = 0.5
        for i in range(n):
            for j in range(n):
                color = "white" if cm_norm[i, j] > thresh else "#333333"
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                        ha="center", va="center", fontsize=7.5, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{title} — Confusion Matrix", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = filename or f"{title.lower().replace(' ', '_')}_confusion.png"
    plt.savefig(fig_path(out), dpi=150, bbox_inches="tight")
    plt.close()
