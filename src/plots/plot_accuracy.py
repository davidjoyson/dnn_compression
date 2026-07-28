import math
import matplotlib.pyplot as plt
from .save_utils import save_fig
from .style import apply_style, METHOD_COLORS, PALETTE


def plot_accuracy(methods, title="Accuracy Comparison", filename="accuracy.png", ylabel="Accuracy"):
    """
    methods: dict of {label: value} or {label: (value, std)}
    First entry is treated as the baseline (Uncompressed) for delta annotations.
    """
    apply_style()

    labels, values = [], []
    for label, entry in methods.items():
        v = entry[0] if isinstance(entry, (list, tuple)) else entry
        if math.isnan(v):
            continue
        labels.append(label)
        values.append(v)

    if not labels:
        return

    colors = [METHOD_COLORS.get(lbl, PALETTE[i % len(PALETTE)]) for i, lbl in enumerate(labels)]
    baseline = values[0]

    fig, ax = plt.subplots(figsize=(max(3.2, len(labels) * 0.9), 5.0))
    ax.grid(False)

    bars = ax.bar(
        labels, values,
        color=colors,
        width=0.55,
        zorder=3,
        edgecolor="white",
        linewidth=0.8,
    )

    # Dashed reference line at uncompressed baseline
    ax.axhline(baseline, color="#666666", linestyle="--", linewidth=1.0, zorder=2, alpha=0.6)

    y_min = max(0.0, min(values) - 0.06)
    y_max = min(1.0, max(values) + 0.10)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=24)
    plt.xticks(rotation=20, ha="right")

    tick_h = (y_max - y_min) * 0.015
    for i, (lbl, v) in enumerate(zip(labels, values)):
        top = v + tick_h
        ax.text(i, top, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        if i > 0:
            delta = v - baseline
            if abs(delta) > 1e-6:
                d_color = "#2CA02C" if delta >= 0 else "#D62728"
                ax.text(i, top + tick_h * 2, f"{delta:+.4f}",
                        ha="center", va="bottom", fontsize=7.5, color=d_color)

    save_fig(filename)
