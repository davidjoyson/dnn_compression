import matplotlib.pyplot as plt
from .save_utils import fig_path

def plot_ablation(results, filename="ablation.png"):
    labels = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(8,4))
    plt.bar(labels, values, color=["gray", "skyblue", "orange", "green"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Ablation Study")

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")

    plt.savefig(fig_path(filename), dpi=300, bbox_inches="tight")
    plt.close()
