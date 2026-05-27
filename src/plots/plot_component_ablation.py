import matplotlib.pyplot as plt
import numpy as np
from .save_utils import fig_path
from .style import apply_style, PALETTE

CONDITION_LABELS = {
    "none":      "Uncompressed",
    "topo_only": "Topo Only",
    "quant_only": "Quant Only",
    "both":      "Both",
}

CONDITION_ORDER = ["none", "topo_only", "quant_only", "both"]


def plot_component_ablation(results, filename="component_ablation.png"):
    apply_style()

    conditions = [c for c in CONDITION_ORDER if c in results]
    labels = [CONDITION_LABELS[c] for c in conditions]
    means  = [results[c]["mean"] for c in conditions]
    stds   = [results[c].get("std", 0.0) for c in conditions]

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(conditions))]
    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.55, zorder=3,
                  edgecolor="white", linewidth=0.8,
                  error_kw=dict(elinewidth=1.2, capsize=4, ecolor="#444444"))

    baseline = means[0]
    ax.axhline(baseline, color="#666666", linestyle="--", linewidth=1.0, zorder=2, alpha=0.6)

    y_min = max(0.0, min(m - s for m, s in zip(means, stds)) - 0.05)
    y_max = min(1.0, max(m + s for m, s in zip(means, stds)) + 0.09)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Compression Component Ablation", pad=14)

    tick_h = (y_max - y_min) * 0.015
    for i, (m, s) in enumerate(zip(means, stds)):
        label = f"{m:.4f}"
        if s > 0:
            label += f"\n±{s:.4f}"
        ax.text(i, m + s + tick_h, label, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=150, bbox_inches="tight")
    plt.close()
