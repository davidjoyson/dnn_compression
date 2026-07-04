import numpy as np
import matplotlib.pyplot as plt
from .save_utils import save_fig
from .style import apply_style


def plot_compression_delta(conf_matrix, class_names, title="", filename=None):
    """Bar chart of per-class recall change (compressed minus uncompressed)."""
    apply_style()

    cm_u = conf_matrix["uncompressed"].astype(float)
    cm_c = conf_matrix["compressed"].astype(float)

    recall_u = np.diag(cm_u) / cm_u.sum(axis=1).clip(min=1)
    recall_c = np.diag(cm_c) / cm_c.sum(axis=1).clip(min=1)
    delta = recall_c - recall_u

    n = len(class_names)
    x = np.arange(n)
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), 4))
    bars = ax.bar(x, delta * 100, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#444444", lw=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Recall Δ (pp)")
    ax.set_title(f"Per-Class Recall Change After Compression{' — ' + title if title else ''}")

    for bar, d in zip(bars, delta):
        va = "bottom" if d >= 0 else "top"
        offset = 0.15 if d >= 0 else -0.15
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{d * 100:+.1f}",
            ha="center", va=va, fontsize=7,
        )

    if filename is None:
        filename = f"{title.lower().replace(' ', '_').replace('/', '_')}_compression_delta.png"
    save_fig(filename)
