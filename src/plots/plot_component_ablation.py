import matplotlib.pyplot as plt
import numpy as np
from .save_utils import save_fig
from .style import apply_style, PALETTE

CONDITION_LABELS = {
    "none":      "Uncompressed",
    "topo_only": "Topo Only",
    "quant_only": "Quant Only",
    "reg_only":  "Reg Only",
    "both":      "Both",
}

CONDITION_ORDER = ["none", "topo_only", "quant_only", "reg_only", "both"]

# Fixed per-condition color, independent of which conditions are present in a
# given chart — "none"/"quant_only" always look the same across every plot.
CONDITION_COLORS = {cond: PALETTE[i % len(PALETTE)] for i, cond in enumerate(CONDITION_ORDER)}

DATASET_LABELS = {"har": "HAR", "ecg": "ECG", "eeg": "EEG", "hapt": "HAPT"}
DATASET_ORDER = ["har", "ecg", "eeg", "hapt"]


def plot_component_ablation(results, filename="component_ablation.png", title="Compression Component Ablation"):
    apply_style()

    conditions = [c for c in CONDITION_ORDER if c in results]
    conditions += [c for c in results if c not in conditions]
    labels = [CONDITION_LABELS.get(c, c.replace("_", " ").title()) for c in conditions]
    means  = [results[c]["mean"] for c in conditions]
    stds   = [results[c].get("std", 0.0) for c in conditions]

    colors = [CONDITION_COLORS.get(c, PALETTE[i % len(PALETTE)]) for i, c in enumerate(conditions)]
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
    ax.set_title(title, pad=14)

    tick_h = (y_max - y_min) * 0.015
    for i, (m, s) in enumerate(zip(means, stds)):
        label = f"{m:.4f}"
        if s > 0:
            label += f"\n±{s:.4f}"
        ax.text(i, m + s + tick_h, label, ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    save_fig(filename)


def plot_ablation_combined(results, filename, title):
    """
    One grouped-bar chart across all datasets for a per-dataset ablation
    result set: {dataset: {condition: {"mean": float, "std": float}}}.
    Datasets on the x-axis, one bar per condition per dataset group.
    """
    apply_style()

    datasets = [d for d in DATASET_ORDER if d in results]
    datasets += [d for d in results if d not in datasets]
    if not datasets:
        return

    conditions = [c for c in CONDITION_ORDER if any(c in results[d] for d in datasets)]

    n_cond = len(conditions)
    group_w = 0.8
    bar_w = group_w / n_cond
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(max(6.5, len(datasets) * 1.9), 4.8))

    for j, cond in enumerate(conditions):
        means = [results[d].get(cond, {}).get("mean", float("nan")) for d in datasets]
        stds  = [results[d].get(cond, {}).get("std", 0.0) for d in datasets]
        offset = (j - (n_cond - 1) / 2) * bar_w
        ax.bar(x + offset, means, yerr=stds, width=bar_w * 0.9,
               color=CONDITION_COLORS.get(cond, PALETTE[j % len(PALETTE)]),
               label=CONDITION_LABELS.get(cond, cond.replace("_", " ").title()),
               zorder=3, edgecolor="white", linewidth=0.8,
               error_kw=dict(elinewidth=1.0, capsize=3, ecolor="#444444"))

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d.upper()) for d in datasets])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=n_cond, frameon=False)

    save_fig(filename)
