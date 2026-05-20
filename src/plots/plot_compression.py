import matplotlib.pyplot as plt
from .save_utils import fig_path
from .style import apply_style, METHOD_COLORS, PALETTE


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
    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=150, bbox_inches="tight")
    plt.close()
