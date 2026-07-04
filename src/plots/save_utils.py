import os
import matplotlib.pyplot as plt

FIG_DIR = "figures"

def set_fig_dir(path):
    global FIG_DIR
    FIG_DIR = path

def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)

def fig_path(name):
    ensure_fig_dir()
    return os.path.join(FIG_DIR, name)

def save_fig(filename):
    plt.tight_layout()
    plt.savefig(fig_path(filename), dpi=150, bbox_inches="tight")
    plt.close()
