import numpy as np
import matplotlib.pyplot as plt
from .save_utils import save_fig
from .style import apply_style, METHOD_COLORS, PALETTE

# The 4 methods highlighted in the README's Edge Deployment section --
# Float32 baseline vs the 3 int8 variants that best illustrate the
# weight-only-vs-true-int8 distinction.
_METHODS = [
    ("Float32 (baseline)",        "Uncompressed",              METHOD_COLORS["Uncompressed"]),
    ("Snowflake int8 (per-layer)", "Snowflake (int8)",          METHOD_COLORS["Snowflake (int8)"]),
    ("Static W+A int8 (FX)",       "Static (int8)",             PALETTE[2]),
    ("Snowflake+Static int8",      "Snowflake+Static (int8)",   METHOD_COLORS["Snowflake+Static (int8)"]),
]

_DATASET_ORDER = ["har", "ecg", "hapt"]
_DATASET_LABELS = {"har": "HAR", "ecg": "ECG", "hapt": "HAPT", "eeg": "EEG"}


def plot_pi_latency(dataframes, filename="pi_latency_batch1.png"):
    """
    Grouped bar chart of real Raspberry Pi 3 batch=1 inference latency
    (ms) across datasets, comparing Float32 baseline against the 3 int8
    variants that best show the weight-only-vs-true-int8 distinction.

    dataframes: {dataset_name: pandas.DataFrame} of benchmark_pi.py's
      output CSV (columns: batch, method, latency_ms, std_ms, ...),
      already filtered or not -- this function filters to batch == 1.
    """
    apply_style()

    datasets = [d for d in _DATASET_ORDER if d in dataframes]
    if not datasets:
        return

    n_ds, n_m = len(datasets), len(_METHODS)
    bar_w = 0.18
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * bar_w
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(7, n_ds * 2.6), 5))

    for i, (csv_method, label, color) in enumerate(_METHODS):
        means, stds = [], []
        for ds in datasets:
            df = dataframes[ds]
            row = df[(df["batch"] == 1) & (df["method"] == csv_method)]
            means.append(float(row["latency_ms"].iloc[0]) if len(row) else np.nan)
            stds.append(float(row["std_ms"].iloc[0]) if len(row) else 0.0)
        bars = ax.bar(x + offsets[i], means, bar_w, label=label, color=color,
                      yerr=stds, capsize=3, zorder=3, edgecolor="white", linewidth=0.6)
        for bar, v, s in zip(bars, means, stds):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + s + 0.15,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Latency (ms, batch=1)")
    ax.set_title("Raspberry Pi 3 Inference Latency — Real Hardware (qnnpack)", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels([_DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8, frameon=False)

    save_fig(filename)
