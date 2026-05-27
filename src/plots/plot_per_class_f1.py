import numpy as np
import matplotlib.pyplot as plt
from .save_utils import fig_path
from .style import apply_style, METHOD_COLORS, PALETTE


def _f1_from_cm(cm):
    """Compute per-class F1 from a confusion matrix (numpy array)."""
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(float(2 * tp / denom) if denom > 0 else 0.0)
    return f1s


def plot_per_class_f1(conf_matrix_dict, class_names, title="", filename=None):
    """
    conf_matrix_dict: {"uncompressed": cm_array, "compressed": cm_array}
    Grouped bar chart showing per-class F1 before and after Snowflake compression.
    """
    apply_style()

    cm_u = conf_matrix_dict.get("uncompressed")
    cm_c = conf_matrix_dict.get("compressed")
    if cm_u is None:
        return

    f1_u = _f1_from_cm(cm_u)
    f1_c = _f1_from_cm(cm_c) if cm_c is not None else None
    n = len(f1_u)
    names = class_names if class_names and len(class_names) == n else [f"Class {i}" for i in range(n)]

    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, n * 1.4), 4.5))

    ax.bar(x - width / 2, f1_u, width, label="Uncompressed",
           color=METHOD_COLORS["Uncompressed"], zorder=3, edgecolor="white")
    if f1_c is not None:
        ax.bar(x + width / 2, f1_c, width, label="Snowflake (int8)",
               color=METHOD_COLORS["Snowflake (int8)"], zorder=3, edgecolor="white")

    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-Class F1{' — ' + title if title else ''}", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()

    tick_h = 0.015
    for i, v in enumerate(f1_u):
        ax.text(i - width / 2, v + tick_h, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    if f1_c is not None:
        for i, v in enumerate(f1_c):
            ax.text(i + width / 2, v + tick_h, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if filename is None:
        slug = title.lower().replace(" ", "_")
        filename = f"{slug}_per_class_f1.png"
    plt.savefig(fig_path(filename), dpi=150, bbox_inches="tight")
    plt.close()
