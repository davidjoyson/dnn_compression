import math
import matplotlib.pyplot as plt
from .save_utils import fig_path


def plot_accuracy(methods, title="Accuracy Comparison", filename="accuracy.png"):
    """
    methods: dict of {label: value} or {label: (value, std)}
    Filters out NaN entries automatically.
    """
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

    colors = [
        "steelblue", "orange", "seagreen", "mediumpurple", "firebrick",
        "darkcyan", "saddlebrown",
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 4))
    bars = ax.bar(labels, values, color=colors[:len(labels)],
                  yerr=[s if s > 0 else None for s in stds] if any(s > 0 for s in stds) else None,
                  capsize=4, error_kw={"elinewidth": 1.2})
    y_min = max(0.0, min(values) - 0.05)
    ax.set_ylim(y_min, min(1.0, max(values) + 0.06))
    ax.set_ylabel("Accuracy")
    ax.set_title(title, pad=12)
    plt.xticks(rotation=15, ha="right")

    for i, (v, s) in enumerate(zip(values, stds)):
        label = f"{v:.3f}"
        if s > 0:
            label += f"\n±{s:.3f}"
        ax.text(i, v + (max(stds) if any(s > 0 for s in stds) else 0) + 0.005,
                label, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=300, bbox_inches="tight")
    plt.close()
