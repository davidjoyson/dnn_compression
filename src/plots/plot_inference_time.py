import matplotlib.pyplot as plt
from .save_utils import fig_path
from .style import apply_style, METHOD_COLORS, PALETTE


def plot_inference_time(all_results, filename="inference_time.png"):
    """
    all_results: dict of {dataset_name: result dict}
    Grouped bar chart of inference time (ms) per method per dataset.
    """
    apply_style()

    method_keys = [
        ("Uncompressed",    "inference_time_uncompressed_ms"),
        ("Snowflake (int8)","inference_time_compressed_ms"),
        ("Dynamic (int8)",  "inference_time_dynamic_ms"),
        ("MLP Baseline",    "inference_time_mlp_ms"),
    ]

    datasets = [ds for ds in all_results if all_results[ds].get("inference_time_uncompressed_ms") is not None]
    if not datasets:
        return

    import numpy as np
    n_ds = len(datasets)
    n_m  = len(method_keys)
    width = 0.18
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(7, n_ds * 2.5), 4.5))

    for i, (label, key) in enumerate(method_keys):
        times = [all_results[ds].get(key) for ds in datasets]
        times = [float(t) if t is not None else 0.0 for t in times]
        color = METHOD_COLORS.get(label, PALETTE[i % len(PALETTE)])
        ax.bar(x + offsets[i], times, width, label=label, color=color,
               zorder=3, edgecolor="white", linewidth=0.6)

    ax.set_ylabel("Inference Time (ms, full test set)")
    ax.set_title("Inference Time by Method and Dataset", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=150, bbox_inches="tight")
    plt.close()
