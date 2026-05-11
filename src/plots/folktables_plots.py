import matplotlib.pyplot as plt
from .save_utils import fig_path


# ============================================================
# 1. Accuracy Comparison (Uncompressed vs Compressed vs MLP)
# ============================================================

def plot_folktables_accuracy(results, filename="folktables_accuracy.png"):
    """
    results = {
        "Dendritic (Uncompressed)": acc,
        "Dendritic (Compressed)": acc,
        "MLP Baseline": acc
    }
    """
    labels = list(results.keys())
    values = list(results.values())

    plt.figure(figsize=(8,5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    bars = plt.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Folktables ACSIncome — Model Accuracy Comparison", fontsize=14)

    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.3f}", ha="center", fontsize=11)

    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=300)
    plt.close()



# ============================================================
# 2. Model Size Comparison (Horizontal Bars)
# ============================================================

def plot_folktables_size(size_uncompressed, size_compressed, filename="folktables_size.png"):
    labels = ["Uncompressed", "Compressed"]
    sizes = [size_uncompressed, size_compressed]

    plt.figure(figsize=(8,4))
    plt.barh(labels, sizes, color=["seagreen", "firebrick"])
    plt.xlabel("Model Size (bytes)")
    plt.title("Folktables Model Size Comparison")

    for i, v in enumerate(sizes):
        plt.text(v + max(sizes)*0.02, i, f"{v:,} bytes", va="center")

    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=300)
    plt.close()



# ============================================================
# 3. Accuracy vs Compression Ratio (Trade-off Curve)
# ============================================================

def plot_folktables_tradeoff(acc_uncompressed, acc_compressed,
                             size_uncompressed, size_compressed,
                             filename="folktables_tradeoff.png"):

    compression_ratio = size_uncompressed / size_compressed

    plt.figure(figsize=(7,5))
    plt.scatter([1, compression_ratio], [acc_uncompressed, acc_compressed],
                s=[120, 120], c=["#4C72B0", "#DD8452"], edgecolor="black")

    plt.plot([1, compression_ratio], [acc_uncompressed, acc_compressed], "--", color="gray")

    plt.xlabel("Compression Ratio", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy vs Compression Ratio (Folktables)", fontsize=14)

    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=300)
    plt.close()

