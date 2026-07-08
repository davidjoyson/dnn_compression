import numpy as np
import matplotlib.pyplot as plt
from .save_utils import save_fig
from .style import apply_style, PALETTE


def plot_branch_diversity(diversity_data, title="", filename="branch_diversity.png"):
    """
    Three-panel figure:
    - Weight cosine similarity heatmap
    - Activation correlation heatmap
    - Per-branch quantization error bar chart
    """
    apply_style()

    weight_sim = diversity_data.get("weight_similarity")
    act_corr   = diversity_data.get("activation_correlation")
    quant_err  = diversity_data.get("quant_error_per_branch")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # --- Weight cosine similarity ---
    ax = axes[0]
    if weight_sim is not None:
        im = ax.imshow(weight_sim, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
        n = weight_sim.shape[0]
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([f"B{i}" for i in range(n)], fontsize=8)
        ax.set_yticklabels([f"B{i}" for i in range(n)], fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{weight_sim[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(weight_sim[i, j]) < 0.6 else "white")
    ax.set_title("Branch Weight\nCosine Similarity")
    ax.set_xlabel("Branch"); ax.set_ylabel("Branch")

    # --- Activation correlation ---
    ax = axes[1]
    if act_corr is not None:
        im = ax.imshow(act_corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
        n = act_corr.shape[0]
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([f"B{i}" for i in range(n)], fontsize=8)
        ax.set_yticklabels([f"B{i}" for i in range(n)], fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{act_corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(act_corr[i, j]) < 0.6 else "white")
    ax.set_title("Branch Activation\nCorrelation")
    ax.set_xlabel("Branch"); ax.set_ylabel("Branch")

    # --- Per-branch quantization error ---
    ax = axes[2]
    if quant_err is not None:
        xs = list(range(len(quant_err)))
        bars = ax.bar(xs, quant_err, color=PALETTE[0], edgecolor="white", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"B{i}" for i in xs], fontsize=8)
        ax.set_xlabel("Branch")
        ax.set_ylabel("MSE")
        # Annotate bars
        for bar, v in zip(bars, quant_err):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2e}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Per-Branch\nQuantization Error (MSE)")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    save_fig(filename)
