import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_xor_boundary(model, save_path=None):
    # Create grid
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200),
                         np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_t = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        preds = model(grid_t).numpy().reshape(xx.shape)

    plt.figure(figsize=(6,6))
    plt.contourf(xx, yy, preds, levels=50, cmap="coolwarm", alpha=0.8)
    plt.colorbar(label="Model Output")

    # XOR points
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    plt.scatter(X[:,0], X[:,1], c=y, cmap="coolwarm", edgecolors="black", s=100)

    plt.title("XOR Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
