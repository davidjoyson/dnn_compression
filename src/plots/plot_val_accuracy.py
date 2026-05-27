import matplotlib.pyplot as plt
from .save_utils import fig_path
from .style import apply_style


_COLORS = {
    "Dendritic": "#4878CF",
    "MLP":       "#E87722",
}


def plot_val_accuracy(val_acc_history, title="", filename=None):
    """
    val_acc_history: {"Dendritic": [acc_per_epoch], "MLP": [acc_per_epoch]}
    Line chart of validation accuracy vs epoch.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, accs in val_acc_history.items():
        if accs is None:
            continue
        color = _COLORS.get(label)
        ax.plot(range(1, len(accs) + 1), accs, label=label, color=color, lw=1.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(f"Validation Accuracy{' — ' + title if title else ''}", pad=12)
    ax.set_ylim(max(0.0, ax.get_ylim()[0] - 0.02), 1.02)
    ax.legend(loc="lower right")

    plt.tight_layout()
    if filename is None:
        slug = title.lower().replace(" ", "_")
        filename = f"{slug}_val_accuracy.png"
    plt.savefig(fig_path(filename), dpi=150, bbox_inches="tight")
    plt.close()
