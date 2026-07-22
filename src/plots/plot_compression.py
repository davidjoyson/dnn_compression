import numpy as np
import matplotlib.pyplot as plt
from .save_utils import save_fig
from .style import apply_style, METHOD_COLORS, PALETTE
from .plot_cross_dataset import METHOD_STUBS, _isnan


def plot_compression(sizes, title="Model Size Comparison", filename="compression.png"):
    """
    sizes: dict of {label: bytes}.
    First entry is the uncompressed baseline; ratios are computed relative to it.
    """
    apply_style()

    labels = list(sizes.keys())
    values = list(sizes.values())

    max_val = max(values)
    if max_val >= 1_000_000:
        scale, unit = 1_000_000, "MB"
    elif max_val >= 1_000:
        scale, unit = 1_000, "KB"
    else:
        scale, unit = 1, "B"
    scaled = [v / scale for v in values]

    colors = [METHOD_COLORS.get(lbl, PALETTE[i % len(PALETTE)]) for i, lbl in enumerate(labels)]
    baseline = scaled[0]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.55), 4.2))
    ax.bar(labels, scaled, color=colors, width=0.55, zorder=3, edgecolor="white", linewidth=0.8)

    ax.set_ylabel(f"Model Size ({unit})")
    ax.set_title(title, pad=14)
    plt.xticks(rotation=20, ha="right")

    tick_h = max(scaled) * 0.018
    for i, (v_sc, v_bytes) in enumerate(zip(scaled, values)):
        top = v_sc + tick_h
        ax.text(i, top, f"{v_bytes:,} B", ha="center", va="bottom", fontsize=8, fontweight="bold")
        if i > 0 and v_sc > 0:
            ratio = baseline / v_sc
            ax.text(i, top + tick_h * 2.2, f"{ratio:.1f}×",
                    ha="center", va="bottom", fontsize=8, color="#2CA02C", fontweight="bold")

    ax.set_ylim(0, max(scaled) * 1.22)
    save_fig(filename)


def plot_compression_by_dataset(all_results, filename="combined/cross_dataset_compression.png"):
    """Grouped bar chart: datasets on X-axis, one bar group per compression method, model size."""
    apply_style()

    datasets = list(all_results.keys())
    methods = [
        (label, "size" + stub)
        for label, stub in METHOD_STUBS
        if any(not _isnan(all_results[d].get("size" + stub)) for d in datasets)
    ]
    if not methods:
        return

    max_val = max(
        float(all_results[ds][key]) for ds in datasets for _, key in methods
        if not _isnan(all_results[ds].get(key))
    )
    scale, unit = (1_000_000, "MB") if max_val >= 1_000_000 else (1_000, "KB") if max_val >= 1_000 else (1, "B")

    n_ds, n_m = len(datasets), len(methods)
    width = min(0.15, 0.8 / n_m)
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(8, n_ds * 3), 5))
    for i, (label, key) in enumerate(methods):
        vals = [
            float(all_results[ds][key]) / scale if not _isnan(all_results[ds].get(key)) else float("nan")
            for ds in datasets
        ]
        color = METHOD_COLORS.get(label, PALETTE[i % len(PALETTE)])
        ax.bar(x + offsets[i], vals, width, label=label, color=color,
               zorder=3, edgecolor="white", linewidth=0.6)

    ax.set_ylabel(f"Model Size ({unit})")
    ax.set_title("Model Size Across Datasets and Compression Methods", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, max_val / scale * 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=8, frameon=False)

    save_fig(filename)
