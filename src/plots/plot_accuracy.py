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

    labels, values, stds = [], [], []
    for label, entry in methods.items():
        if isinstance(entry, (list, tuple)):
            v, s = entry[0], entry[1]
        else:
            v, s = entry, 0.0
        if math.isnan(v):
            continue
        labels.append(label)
        values.append(v)
        stds.append(s)

    if not labels:
        return

    colors = [METHOD_COLORS.get(lbl, PALETTE[i % len(PALETTE)]) for i, lbl in enumerate(labels)]
    has_err = any(s > 0 for s in stds)
    baseline = values[0]

    fig, ax = plt.subplots(figsize=(max(5.5, len(labels) * 1.55), 4.8))

    bars = ax.bar(
        labels, values,
        color=colors,
        yerr=[s if s > 0 else float("nan") for s in stds] if has_err else None,
        capsize=5,
        error_kw={"elinewidth": 1.5, "ecolor": "#555555", "capthick": 1.5},
        width=0.55,
        zorder=3,
        edgecolor="white",
        linewidth=0.8,
    )

    # Dashed reference line at uncompressed baseline
    ax.axhline(baseline, color="#666666", linestyle="--", linewidth=1.0, zorder=2, alpha=0.6)

    err_top = max(stds) if has_err else 0.0
    y_min = max(0.0, min(values) - 0.06)
    y_max = min(1.0, max(values) + err_top + 0.10)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=24)
    plt.xticks(rotation=20, ha="right")

    tick_h = (y_max - y_min) * 0.015
    for i, (lbl, v, s) in enumerate(zip(labels, values, stds)):
        top = v + (s if s > 0 else 0) + tick_h
        ax.text(i, top, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        if i > 0:
            delta = v - baseline
            if abs(delta) > 1e-6:
                d_color = "#2CA02C" if delta >= 0 else "#D62728"
                ax.text(i, top + tick_h * 2, f"{delta:+.4f}",
                        ha="center", va="bottom", fontsize=7.5, color=d_color)

    save_fig(filename)
