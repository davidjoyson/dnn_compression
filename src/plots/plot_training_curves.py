import matplotlib.pyplot as plt
from .save_utils import save_fig
from .style import apply_style


_TRAIN_COLORS = {
    "Dendritic (Uncompressed)": "#4878CF",
    "MLP Baseline":             "#E87722",
    "Dendritic (Val)":          "#4878CF",
    "MLP (Val)":                "#E87722",
}


def _smooth(values, alpha=0.6):
    out, s = [], None
    for v in values:
        s = v if s is None else alpha * s + (1 - alpha) * v
        out.append(s)
    return out


def plot_training_curves(histories, title="", filename=None):
    """
    histories: dict of {label: [loss_per_epoch]}
    Labels containing "(Val)" are drawn dashed; raw values are EMA-smoothed.
    """
    apply_style()

    plt.figure(figsize=(7, 4))
    for label, losses in histories.items():
        color = _TRAIN_COLORS.get(label)
        is_val = "(Val)" in label
        smoothed = _smooth(losses, alpha=0.7)
        plt.plot(range(1, len(smoothed) + 1), smoothed,
                 label=label, color=color, lw=1.8,
                 linestyle="--" if is_val else "-",
                 alpha=0.65 if is_val else 1.0)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curves{' — ' + title if title else ''}", pad=12)
    plt.legend(loc="upper right")
    if filename is None:
        filename = f"{title.lower().replace(' ', '_').replace('/', '_')}_training_curves.png"
    save_fig(filename)
