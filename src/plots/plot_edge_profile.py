import numpy as np
import matplotlib.pyplot as plt
from .save_utils import save_fig
from .style import apply_style, METHOD_COLORS, PALETTE


def plot_edge_profile(all_results, filename="combined/edge_profile.png"):
    """
    3-panel edge-AI summary across datasets:
      1. Model size (KB) — uncompressed vs Snowflake, with compression ratio annotated
      2. Per-sample latency (μs) — all four methods grouped by dataset
      3. Inference throughput (K samples/sec) — all four methods grouped by dataset
    """
    apply_style()

    datasets = [
        ds for ds, r in all_results.items()
        if isinstance(r, dict)
        and isinstance(r.get("edge_profile"), dict)
        and r["edge_profile"].get("model_size_kb") is not None
    ]
    if not datasets:
        return

    n_ds = len(datasets)
    method_keys = [
        ("Uncompressed",     "uncompressed"),
        ("Snowflake (int8)", "compressed"),
        ("Dynamic (int8)",   "dynamic"),
        ("MLP Baseline",     "mlp"),
    ]
    n_m   = len(method_keys)
    bar_w = 0.16
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * bar_w
    x = np.arange(n_ds)
    xlabels = [d.replace(" ", "\n") for d in datasets]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # ---- Panel 1: Model size (KB) ----
    ax = axes[0]
    sizes_u = [all_results[ds]["edge_profile"]["model_size_kb"] or 0 for ds in datasets]
    sizes_c = [all_results[ds]["edge_profile"].get("compressed_size_kb") or 0 for ds in datasets]
    w2 = bar_w * 1.3
    ax.bar(x - w2 / 2, sizes_u, w2, label="Uncompressed",
           color=METHOD_COLORS["Uncompressed"], zorder=3, edgecolor="white", linewidth=0.6)
    ax.bar(x + w2 / 2, sizes_c, w2, label="Snowflake (int8)",
           color=METHOD_COLORS["Snowflake (int8)"], zorder=3, edgecolor="white", linewidth=0.6)
    for i, ds in enumerate(datasets):
        cr = all_results[ds]["edge_profile"].get("compression_ratio")
        if cr:
            ax.text(i, sizes_u[i] * 1.04, f"{cr:.1f}×",
                    ha="center", va="bottom", fontsize=7, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Model Size (KB)")
    ax.set_title("Model Size")
    ax.legend(fontsize=7)

    # ---- Panel 2: Per-sample latency (μs) ----
    ax = axes[1]
    for i, (label, key) in enumerate(method_keys):
        vals = [
            (all_results[ds]["edge_profile"]["latency_us"].get(key) or 0)
            for ds in datasets
        ]
        color = METHOD_COLORS.get(label, PALETTE[i % len(PALETTE)])
        ax.bar(x + offsets[i], vals, bar_w, label=label,
               color=color, zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Latency per sample (μs)")
    ax.set_title("Per-Sample Latency")
    ax.legend(fontsize=7)

    # ---- Panel 3: Throughput (K samples/sec) ----
    ax = axes[2]
    for i, (label, key) in enumerate(method_keys):
        vals = [
            (all_results[ds]["edge_profile"]["throughput_sps"].get(key) or 0) / 1_000
            for ds in datasets
        ]
        color = METHOD_COLORS.get(label, PALETTE[i % len(PALETTE)])
        ax.bar(x + offsets[i], vals, bar_w, label=label,
               color=color, zorder=3, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Throughput (K samples/sec)")
    ax.set_title("Inference Throughput")
    ax.legend(fontsize=7)

    save_fig(filename)
