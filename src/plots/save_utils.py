import os

FIG_DIR = "figures"

def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)

def fig_path(name):
    ensure_fig_dir()
    return os.path.join(FIG_DIR, name)
