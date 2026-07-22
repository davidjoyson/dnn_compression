import math
import numpy as np
import matplotlib.pyplot as plt
from .save_utils import save_fig
from .style import apply_style, METHOD_COLORS, PALETTE

# Fixed method order + label -> result-key stub, shared by accuracy and F1
# (e.g. "accuracy" + stub / "std" + stub, or "f1" + stub / "std_f1" + stub).
METHOD_STUBS = [
    ("Uncompressed",             "_uncompressed"),
    ("Snowflake (int8)",         "_compressed"),
    ("Global int8",              "_compressed_global"),
    ("Dynamic (int8)",           "_compressed_dynamic"),
    ("Static (int8)",            "_compressed_static"),
    ("Snowflake+Static (int8)",  "_compressed_snowflake_static"),
    ("Per-channel (int8)",       "_compressed_perchan"),
    ("QAT (int8)",               "_compressed_qat"),
    ("Mixed precision",          "_compressed_mixed"),
    ("Snowflake (int4)",         "_compressed_int4"),
    ("MLP Baseline",             "_mlp_baseline"),
]


def _isnan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _plot_grouped_metric(all_results, key_prefix, std_prefix, title, filename, ylabel):
    apply_style()

    datasets = list(all_results.keys())
    methods = [
        (label, key_prefix + stub, std_prefix + stub)
        for label, stub in METHOD_STUBS
        if any(not _isnan(all_results[d].get(key_prefix + stub)) for d in datasets)
    ]
    if not methods:
        return

    n_ds, n_m = len(datasets), len(methods)
    width = min(0.15, 0.8 / n_m)
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(8, n_ds * 3), 5))

    for i, (label, acc_key, std_key) in enumerate(methods):
        vals, stds = [], []
        for ds in datasets:
            r = all_results[ds]
            v = r.get(acc_key)
            s = r.get(std_key, 0.0)
            vals.append(float(v) if not _isnan(v) else float("nan"))
            stds.append(float(s) if s == s else 0.0)

        color = METHOD_COLORS.get(label, PALETTE[i % len(PALETTE)])
        ax.bar(x + offsets[i], vals, width, label=label, color=color,
               yerr=stds, capsize=3, zorder=3, edgecolor="white", linewidth=0.6)

    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize=8, frameon=False)
    all_vals = [
        float(all_results[ds][key])
        for ds in datasets
        for _, key, _ in methods
        if not _isnan(all_results[ds].get(key))
    ]
    ymin = max(0.0, min(all_vals) - 0.08) if all_vals else 0.0
    ax.set_ylim(ymin, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    save_fig(filename)


def plot_cross_dataset_summary(all_results, filename="combined/cross_dataset_summary.png"):
    """Grouped bar chart: datasets on X-axis, one bar group per compression method, accuracy."""
    _plot_grouped_metric(all_results, "accuracy", "std",
                          "Accuracy Across Datasets and Compression Methods", filename, "Accuracy")


def plot_cross_dataset_f1(all_results, filename="combined/cross_dataset_f1.png"):
    """Grouped bar chart: datasets on X-axis, one bar group per compression method, macro F1."""
    _plot_grouped_metric(all_results, "f1", "std_f1",
                          "Macro F1 Across Datasets and Compression Methods", filename, "Macro F1")
