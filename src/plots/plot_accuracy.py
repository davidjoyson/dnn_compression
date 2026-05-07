import matplotlib.pyplot as plt
from .save_utils import fig_path

def plot_accuracy(acc_uncompressed, acc_compressed, title="Accuracy Comparison", filename="accuracy.png"):
    labels = ["Uncompressed", "Compressed"]
    values = [acc_uncompressed, acc_compressed]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values, color=["steelblue", "orange"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(title)

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")

    plt.savefig(fig_path(filename), dpi=300, bbox_inches="tight")
    plt.close()
