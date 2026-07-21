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
_DATASET_COLORS = {"har": PALETTE[0], "ecg": PALETTE[1], "hapt": PALETTE[2], "eeg": PALETTE[3]}

# Fixed color assignment for all 10 methods in the CSV, reused across every
# plot in this module -- extends METHOD_COLORS with 3 methods that have no
# existing entry (int4, Static, QAT), never cycled/reassigned between plots.
_ALL_METHODS = [
    ("Float32 (baseline)",         "Float32",          METHOD_COLORS["Uncompressed"]),
    ("Snowflake int8 (per-layer)", "Snowflake",        METHOD_COLORS["Snowflake (int8)"]),
    ("Global int8",                "Global",           METHOD_COLORS["Global int8"]),
    ("Per-channel int8",           "Per-channel",      METHOD_COLORS["Dynamic (int8)"]),
    ("Snowflake int4 (per-layer)", "Snowflake int4",   "#BCBD22"),
    ("Dynamic int8 (qnnpack)",     "Dynamic",          METHOD_COLORS["MLP Baseline"]),
    ("Static W+A int8 (FX)",       "Static W+A",       "#7F7F7F"),
    ("Snowflake+Static int8",      "Snowflake+Static", METHOD_COLORS["Snowflake+Static (int8)"]),
    ("Mixed precision (FX)",       "Mixed precision",  METHOD_COLORS["MLP Compressed"]),
    ("QAT int8 (FX)",              "QAT",              "#E377C2"),
]


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


def plot_pi_memory(dataframes, filename="pi_memory_batch1.png"):
    """
    Grouped bar chart of real Raspberry Pi 3 process memory (RSS, MB) at
    batch=1, same 4 methods as plot_pi_latency() for a direct side-by-side
    read: the true-int8 methods (Static/Snowflake+Static) are faster but
    use more memory than the weight-only methods (Float32/Snowflake).
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
        means = []
        for ds in datasets:
            df = dataframes[ds]
            row = df[(df["batch"] == 1) & (df["method"] == csv_method)]
            means.append(float(row["rss_mb"].iloc[0]) if len(row) else np.nan)
        bars = ax.bar(x + offsets[i], means, bar_w, label=label, color=color,
                      zorder=3, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, means):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 3,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Process memory, RSS (MB, batch=1)")
    ax.set_title("Raspberry Pi 3 Memory Usage — Real Hardware", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels([_DATASET_LABELS.get(d, d) for d in datasets])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8, frameon=False)

    save_fig(filename)


def plot_pi_speedup_all_methods(dataframes, filename="pi_speedup_all_methods.png"):
    """
    One horizontal-bar panel per dataset, all 10 methods, real-hardware
    speedup at batch=1 relative to Float32 -- a reference line at 1.0
    makes "faster than baseline" vs "no benefit" immediately visible for
    every method at once, not just the 4 highlighted in the main chart.
    """
    apply_style()

    datasets = [d for d in _DATASET_ORDER if d in dataframes]
    if not datasets:
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), sharex=True)
    if len(datasets) == 1:
        axes = [axes]

    y = np.arange(len(_ALL_METHODS))
    for ax, ds in zip(axes, datasets):
        df = dataframes[ds]
        speedups, colors = [], []
        for csv_method, label, color in _ALL_METHODS:
            row = df[(df["batch"] == 1) & (df["method"] == csv_method)]
            speedups.append(float(row["speedup"].iloc[0]) if len(row) else np.nan)
            colors.append(color)
        ax.barh(y, speedups, color=colors, zorder=3, edgecolor="white", linewidth=0.6)
        ax.axvline(1.0, color="#333333", linewidth=1, linestyle="--", zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([label for _, label, _ in _ALL_METHODS], fontsize=8)
        ax.set_xlabel("Speedup vs Float32 (batch=1)")
        ax.set_title(_DATASET_LABELS.get(ds, ds))
        for i, v in enumerate(speedups):
            if not np.isnan(v):
                ax.text(v + 0.03, i, f"{v:.2f}×", va="center", fontsize=7)

    fig.suptitle("Real-Hardware Speedup by Method — Raspberry Pi 3", fontsize=13, fontweight="bold", y=1.02)
    save_fig(filename)


def plot_pi_batch_comparison(dataframes, filename="pi_batch_comparison.png"):
    """
    Two-panel horizontal bar chart: real-hardware speedup at batch=-1
    (full-throughput) vs batch=1 (single-sample), all 10 methods, colored
    by dataset. Side-by-side panels surface methods whose speedup verdict
    flips between batch modes (e.g. Dynamic quantization is slower than
    baseline at batch=-1 but faster at batch=1).
    """
    apply_style()

    datasets = [d for d in _DATASET_ORDER if d in dataframes]
    if not datasets:
        return

    n_m, n_ds = len(_ALL_METHODS), len(datasets)
    bar_h = 0.8 / n_ds
    offsets = np.linspace(-(n_ds - 1) / 2, (n_ds - 1) / 2, n_ds) * bar_h
    y = np.arange(n_m)

    fig, (ax_full, ax_one) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    for ax, batch_val, title in [(ax_full, -1, "batch=-1 (full-throughput)"), (ax_one, 1, "batch=1 (single-sample)")]:
        for i, ds in enumerate(datasets):
            df = dataframes[ds]
            speedups = []
            for csv_method, _, _ in _ALL_METHODS:
                row = df[(df["batch"] == batch_val) & (df["method"] == csv_method)]
                speedups.append(float(row["speedup"].iloc[0]) if len(row) else np.nan)
            ax.barh(y + offsets[i], speedups, bar_h, label=_DATASET_LABELS.get(ds, ds),
                    color=_DATASET_COLORS.get(ds, PALETTE[i]), zorder=3, edgecolor="white", linewidth=0.5)
        ax.axvline(1.0, color="#333333", linewidth=1, linestyle="--", zorder=2)
        ax.set_title(title)
        ax.set_xlabel("Speedup vs Float32")

    ax_full.set_yticks(y)
    ax_full.set_yticklabels([label for _, label, _ in _ALL_METHODS], fontsize=8)
    ax_one.legend(loc="lower right", fontsize=8)

    fig.suptitle("Speedup Verdict Can Flip Between Batch Modes — Raspberry Pi 3", fontsize=13, fontweight="bold", y=1.02)
    save_fig(filename)


def plot_pi_pareto(dataframes, filename="pi_pareto_real.png"):
    """
    Scatter: real-hardware compression ratio vs speedup (batch=1), color
    = method (fixed, matches every other Pi plot), marker shape = dataset.
    Same spirit as the existing (simulated) pareto_compression.png, but
    built from actual Pi measurements instead of an estimated edge profile.
    """
    apply_style()

    datasets = [d for d in _DATASET_ORDER if d in dataframes]
    if not datasets:
        return

    markers = {"har": "o", "ecg": "s", "hapt": "D", "eeg": "^"}
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for ds in datasets:
        df = dataframes[ds]
        marker = markers.get(ds, "o")
        for csv_method, label, color in _ALL_METHODS:
            row = df[(df["batch"] == 1) & (df["method"] == csv_method)]
            if not len(row):
                continue
            ax.scatter(float(row["compression"].iloc[0]), float(row["speedup"].iloc[0]),
                      color=color, marker=marker, s=80, zorder=4, edgecolors="white", linewidths=0.6)

    ax.axhline(1.0, color="#333333", linewidth=1, linestyle="--", zorder=2)
    method_patches = [plt.Line2D([0], [0], marker="o", color=c, linestyle="None", markersize=7, label=l)
                      for _, l, c in _ALL_METHODS]
    ds_lines = [plt.Line2D([0], [0], marker=markers.get(ds, "o"), color="grey", linestyle="None",
                           markersize=7, label=_DATASET_LABELS.get(ds, ds)) for ds in datasets]
    legend1 = ax.legend(handles=method_patches, fontsize=7, loc="upper left", title="Method", ncol=2)
    ax.add_artist(legend1)
    ax.legend(handles=ds_lines, fontsize=8, loc="lower right", title="Dataset")

    ax.set_xlabel("Compression ratio (size, ×)")
    ax.set_ylabel("Real speedup vs Float32 (batch=1)")
    ax.set_title("Compression vs Real Speedup — Raspberry Pi 3", pad=14)

    save_fig(filename)
