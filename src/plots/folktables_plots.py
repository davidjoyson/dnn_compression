import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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



# ============================================================
# 4. Feature Sensitivity Heatmap
# ============================================================

def plot_folktables_feature_sensitivity(importances, feature_names,
                                        filename="folktables_feature_sensitivity.png"):

    plt.figure(figsize=(10,6))
    sns.heatmap(importances.reshape(1, -1), cmap="viridis",
                xticklabels=feature_names, yticklabels=["Importance"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Sensitivity — Uncompressed Dendritic Model")
    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=300)
    plt.close()



# ============================================================
# 5. Partial Topology Sharing Curve
# ============================================================

def plot_partial_sharing(sharing_levels, accuracies, filename="folktables_partial_sharing.png"):
    plt.figure(figsize=(8,5))
    plt.plot(sharing_levels, accuracies, marker="o", linewidth=2, color="#4C72B0")

    plt.xlabel("Percentage of Shared Branch Weights")
    plt.ylabel("Accuracy")
    plt.title("Effect of Partial Topology Sharing on Folktables")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=300)
    plt.close()



# ============================================================
# 6. Distribution Shift (Year-to-Year)
# ============================================================

def plot_folktables_shift(results, filename="folktables_shift.png"):
    """
    results = {
        "Uncompressed": {"2016": acc, "2018": acc},
        "Compressed": {"2016": acc, "2018": acc}
    }
    """

    models = list(results.keys())
    years = list(results[models[0]].keys())

    x = range(len(years))
    width = 0.35

    plt.figure(figsize=(8,5))

    for i, model in enumerate(models):
        accs = [results[model][year] for year in years]
        plt.bar([p + i*width for p in x], accs, width=width, label=model)

    plt.xticks([p + width/2 for p in x], years)
    plt.ylabel("Accuracy")
    plt.title("Folktables Distribution Shift (2016 → 2018)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=300)
    plt.close()



# ============================================================
# 7. State Generalization Heatmap
# ============================================================

def plot_folktables_states(state_matrix, train_states, test_states,
                           filename="folktables_state_heatmap.png"):

    plt.figure(figsize=(10,6))
    sns.heatmap(state_matrix, cmap="viridis", annot=True, fmt=".3f")

    plt.xticks(range(len(test_states)), test_states, rotation=45)
    plt.yticks(range(len(train_states)), train_states)

    plt.title("State Generalization Heatmap (Folktables)")
    plt.xlabel("Test State")
    plt.ylabel("Train State")

    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=300)
    plt.close()
