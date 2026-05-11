import matplotlib.pyplot as plt
import numpy as np
from .save_utils import fig_path


def plot_folktables_multistate(results, filename="folktables_multistate.png"):
    """
    Grouped bar chart: uncompressed vs compressed accuracy across test states.
    results: return value of run_folktables_multistate()
    """
    states   = results["test_states"]
    acc_u    = results["accuracy_uncompressed"]
    acc_c    = results["accuracy_compressed"]
    train_st = results["train_state"]

    x     = np.arange(len(states))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_u = ax.bar(x - width / 2, acc_u, width, label="Uncompressed", color="#4C72B0", edgecolor="black")
    bars_c = ax.bar(x + width / 2, acc_c, width, label="Compressed",   color="#DD8452", edgecolor="black")

    for bar in bars_u:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_c:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Test State")
    ax.set_title(f"Folktables Multi-State Generalisation (trained on {train_st})")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=150)
    plt.close()
