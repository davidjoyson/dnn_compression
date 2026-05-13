import os

FIG_DIR = "figures"

def set_fig_dir(path):
    global FIG_DIR
    FIG_DIR = path

def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)

def fig_path(name):
    ensure_fig_dir()
    return os.path.join(FIG_DIR, name)
