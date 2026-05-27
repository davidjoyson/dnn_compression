import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from .save_utils import fig_path
from .style import apply_style, METHOD_COLORS, PALETTE

_DS_MARKERS = {
    "UCI HAR":       "o",
    "ECG Heartbeat": "s",
    "EEG Brainwave": "^",
}

_METHOD_KEYS = [
    ("Uncompressed",    "size_uncompressed",       "accuracy_uncompressed"),
    ("Snowflake (int8)","size_compressed",          "accuracy_compressed"),
    ("Global int8",     "size_compressed_global",   "accuracy_compressed_global"),
    ("Dynamic (int8)",  "size_compressed_dynamic",  "accuracy_compressed_dynamic"),
    ("MLP Baseline",    "size_mlp_uncompressed",    "accuracy_mlp_baseline"),
]


def plot_pareto(all_results, filename="pareto_compression.png"):
    """
    Scatter plot: X = model size (KB), Y = accuracy.
    Color encodes compression method; marker shape encodes dataset.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for ds_idx, (ds_name, r) in enumerate(all_results.items()):
        marker = _DS_MARKERS.get(ds_name, "D")
        for i, (method, size_key, acc_key) in enumerate(_METHOD_KEYS):
            size = r.get(size_key)
            acc  = r.get(acc_key)
            if size is None or acc is None:
                continue
            try:
                size_kb = float(size) / 1000
                acc_f   = float(acc)
            except (TypeError, ValueError):
                continue
            color = METHOD_COLORS.get(method, PALETTE[i % len(PALETTE)])
            ax.scatter(size_kb, acc_f, color=color, marker=marker, s=90,
                       zorder=4, edgecolors="white", linewidths=0.6)
            ax.annotate(ds_name[:3], (size_kb, acc_f),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color=color, fontweight="bold")

    # Legend: methods by color
    method_patches = [
        mpatches.Patch(color=METHOD_COLORS.get(m, PALETTE[i % len(PALETTE)]), label=m)
        for i, (m, _, _) in enumerate(_METHOD_KEYS)
    ]
    # Legend: datasets by marker
    ds_lines = [
        plt.Line2D([0], [0], marker=mk, color="grey", linestyle="None",
                   markersize=7, label=ds)
        for ds, mk in _DS_MARKERS.items()
    ]
    legend1 = ax.legend(handles=method_patches, fontsize=8, loc="lower right")
    ax.add_artist(legend1)
    ax.legend(handles=ds_lines, fontsize=8, loc="lower left")

    ax.set_xlabel("Model Size (KB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Compression–Accuracy Trade-off (Pareto)", pad=14)
    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=150, bbox_inches="tight")
    plt.close()
