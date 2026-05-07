import matplotlib.pyplot as plt
import numpy as np

def plot_scaling(results, neurons1_list, neurons2_list, branches_list,
                 branch_index=0, filename="scaling_heatmap.png"):

    """
    results: 3D array [len(neurons1_list), len(neurons2_list), len(branches_list)]
    branch_index: which branch count to visualize
    """

    heatmap = results[:, :, branch_index]

    plt.figure(figsize=(8,6))
    plt.imshow(heatmap, cmap="viridis", origin="lower")
    plt.colorbar(label="Accuracy")

    plt.xticks(range(len(neurons2_list)), neurons2_list)
    plt.yticks(range(len(neurons1_list)), neurons1_list)

    plt.xlabel("hidden_neurons2")
    plt.ylabel("hidden_neurons1")
    plt.title(f"Scaling Heatmap (branches={branches_list[branch_index]})")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
