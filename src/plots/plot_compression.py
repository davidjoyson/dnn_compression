import matplotlib.pyplot as plt
from .save_utils import fig_path

def plot_compression(size_uncompressed, size_compressed, filename="compression.png"):
    labels = ["Uncompressed", "Compressed"]
    values = [size_uncompressed, size_compressed]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values, color=["seagreen", "firebrick"])
    plt.ylabel("Model Size (bytes)")
    plt.title("Model Compression")

    for i, v in enumerate(values):
        plt.text(i, v + max(values)*0.05, f"{v} bytes", ha="center")

    plt.savefig(fig_path(filename), dpi=300, bbox_inches="tight")
    plt.close()
