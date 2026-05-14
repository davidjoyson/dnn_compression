import matplotlib.pyplot as plt
from .save_utils import fig_path

_COLORS = {
    "Dendritic (Uncompressed)": "#4C72B0",
    "MLP Baseline":             "#DD8452",
    "Dendritic (Val)":          "#4C72B0",
    "MLP (Val)":                "#DD8452",
}


def plot_training_curves(histories, title="", filename=None):
    """
    histories: dict of {label: [avg_loss_per_epoch]}
    Val entries (label contains "(Val)") are drawn dashed at reduced opacity.
    """
    plt.figure(figsize=(7, 4))
    for label, losses in histories.items():
        color = _COLORS.get(label)
        is_val = "(Val)" in label
        plt.plot(range(1, len(losses) + 1), losses, label=label, color=color, lw=1.5,
                 linestyle="--" if is_val else "-", alpha=0.6 if is_val else 1.0)
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title(f"Training Loss{' — ' + title if title else ''}", pad=12)
    plt.legend()
    plt.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    if filename is None:
        slug = title.lower().replace(" ", "_").replace("/", "_")
        filename = f"{slug}_training_curves.png"
    plt.savefig(fig_path(filename), dpi=150)
    plt.close()
