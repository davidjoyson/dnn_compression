import matplotlib.pyplot as plt
from .save_utils import save_fig
from .style import apply_style, PALETTE


def plot_ablation(results, filename="ablation.png"):
    apply_style()

    items = sorted(results.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    baseline = values[0]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4.2))
    ax.bar(labels, values, color=colors, width=0.55, zorder=3, edgecolor="white", linewidth=0.8)

    ax.axhline(baseline, color="#666666", linestyle="--", linewidth=1.0, zorder=2, alpha=0.6)

    y_min = max(0.0, min(values) - 0.05)
    y_max = min(1.0, max(values) + 0.09)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Accuracy")
    ax.set_title("Ablation Study", pad=14)
    plt.xticks(rotation=20, ha="right")

    tick_h = (y_max - y_min) * 0.015
    for i, v in enumerate(values):
        ax.text(i, v + tick_h, f"{v:.4f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")

    save_fig(filename)
