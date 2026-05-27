import math
import numpy as np
import matplotlib.pyplot as plt
from .save_utils import fig_path
from .style import apply_style, METHOD_COLORS, PALETTE


def plot_cross_dataset_summary(all_results, filename="cross_dataset_summary.png"):
    """
    all_results: dict of {dataset_name: flattened result dict (from store_simple)}
    Grouped bar chart: datasets on X-axis, one bar group per method.
    """
    apply_style()

    methods = [
        ("Uncompressed",    "accuracy_uncompressed",        "std_uncompressed"),
        ("Snowflake (int8)","accuracy_compressed",          "std_compressed"),
        ("Global int8",     "accuracy_compressed_global",   "std_compressed_global"),
        ("Dynamic (int8)",  "accuracy_compressed_dynamic",  "std_compressed_dynamic"),
        ("MLP Baseline",    "accuracy_mlp_baseline",        "std_mlp_baseline"),
    ]

    datasets = list(all_results.keys())
    n_ds = len(datasets)
    n_m  = len(methods)
    width = 0.15
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(8, n_ds * 3), 5))

    for i, (label, acc_key, std_key) in enumerate(methods):
        accs, stds = [], []
        for ds in datasets:
            r = all_results[ds]
            v = r.get(acc_key, float("nan"))
            s = r.get(std_key, 0.0)
            accs.append(float(v) if not (isinstance(v, float) and math.isnan(v)) else float("nan"))
            stds.append(float(s) if s == s else 0.0)

        color = METHOD_COLORS.get(label, PALETTE[i % len(PALETTE)])
        ax.bar(x + offsets[i], accs, width, label=label, color=color,
               yerr=stds, capsize=3, zorder=3, edgecolor="white", linewidth=0.6)

    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Across Datasets and Compression Methods", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc="lower right", fontsize=8)
    ymin = max(0.0, min(
        float(all_results[ds].get("accuracy_compressed_dynamic", 1.0))
        for ds in datasets
    ) - 0.05)
    ax.set_ylim(ymin, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=150, bbox_inches="tight")
    plt.close()
