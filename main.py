import argparse
import time

import torch
from tqdm import tqdm

from src.experiments.uci_adult_experiment import run_uci_adult_income
from src.experiments.folktables_experiment import run_folktables_income
from src.experiments.ablation_study import run_ablation, run_compression_component_ablation
from src.experiments.scaling_experiment import run_scaling_experiment
from src.experiments.har_experiment import run_har

from src.loaders.load_adult import load_adult_income
from src.loaders.load_wine import load_wine

from src.models.dendritic_network import DendriticNetwork
from src.models.mlp_baseline import MLPBaseline

import src.plots.save_utils as _save_utils

from src.reporting import (
    store_simple, make_run_dir,
    save_metrics_csv, save_summary_txt,
    print_summary, generate_plots,
)

# ------------------------------------------------------------------ #
# Defaults                                                            #
# ------------------------------------------------------------------ #

SEEDS  = (42, 0, 7)
EPOCHS = 50

ALL_EXPERIMENTS = [
    # "adult", "folktables",
    "ablation", "component",  # "scaling",
    "har",
]

# ------------------------------------------------------------------ #
# Experiment runners (each fills results + timings)                   #
# ------------------------------------------------------------------ #

def _run_adult(results, timings, epochs, seeds, fine_tune_epochs=3):
    print("\n=== UCI Adult Income ===\n")
    t0 = time.time()
    store_simple(results, timings, "UCI Adult Income",
                 run_uci_adult_income(epochs=epochs, seeds=seeds, fine_tune_epochs=fine_tune_epochs),
                 time.time() - t0)


def _run_folktables(results, timings, epochs, seeds):
    print("\n=== Folktables CA 2018 ===\n")
    t0 = time.time()
    store_simple(results, timings, "Folktables CA 2018",
                 run_folktables_income("CA", 2018, epochs=epochs),
                 time.time() - t0)


def _run_ablation(results, timings, epochs, seeds):
    print("\n=== Ablation Study ===\n")
    X_tr, y_tr, X_te, y_te = load_wine()
    configs = [
        {"h1": 16, "h2":  8, "branches": 2, "hidden_per_branch": 4},
        {"h1": 32, "h2": 16, "branches": 4, "hidden_per_branch": 4},
        {"h1": 64, "h2": 32, "branches": 6, "hidden_per_branch": 4},
    ]
    t0 = time.time()
    results["Ablation Study"] = run_ablation(
        configs=configs, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te, epochs=epochs,
    )
    timings["Ablation Study"] = time.time() - t0


def _run_component(results, timings, epochs, seeds):
    print("\n=== Component Ablation ===\n")
    X_tr, y_tr, X_te, y_te = load_wine()
    t0 = time.time()
    results["Component Ablation"] = run_compression_component_ablation(
        X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        config={"h1": 32, "h2": 16, "branches": 4, "hidden_per_branch": 4},
        epochs=epochs, seeds=seeds,
    )
    timings["Component Ablation"] = time.time() - t0


def _run_scaling(results, timings, epochs, seeds):
    print("\n=== Scaling Experiment ===\n")
    X_raw, y_raw = load_adult_income()
    X = torch.tensor(X_raw, dtype=torch.float32)
    y = torch.tensor(y_raw, dtype=torch.float32).reshape(-1, 1)
    split = int(0.8 * len(X))
    t0 = time.time()
    results["Scaling Experiment"] = run_scaling_experiment(
        X_train=X[:split], y_train=y[:split], X_test=X[split:], y_test=y[split:],
        neurons1_list=[16, 32], neurons2_list=[8, 16], branches_list=[2, 4],
        hidden_per_branch=4, epochs=epochs,
    )
    timings["Scaling Experiment"] = time.time() - t0


def _run_har(results, timings, epochs, seeds, fine_tune_epochs=3):
    print("\n=== UCI HAR (Wearable Sensors) ===\n")
    t0 = time.time()
    store_simple(results, timings, "UCI HAR",
                 run_har(epochs=epochs, seeds=seeds, fine_tune_epochs=fine_tune_epochs),
                 time.time() - t0)


REGISTRY = {
    "adult":      _run_adult,
    "folktables": _run_folktables,
    "ablation":   _run_ablation,
    "component":  _run_component,
    "scaling":     _run_scaling,
    "har":         _run_har,
}

# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="DNN Compression Experiments",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--exp", nargs="+", choices=ALL_EXPERIMENTS, default=ALL_EXPERIMENTS, metavar="EXP",
        help="Experiments to run (default: all).\nChoices: " + ", ".join(ALL_EXPERIMENTS),
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Training epochs per experiment (default: {EPOCHS})")
    parser.add_argument("--fine-tune-epochs", type=int, default=3,
                        help="Post-quantization fine-tuning epochs for adult/har (default: 3)")
    parser.add_argument("--arch", action="store_true", help="Print model architectures and exit")
    args = parser.parse_args()

    if args.arch:
        import sys
        from torchinfo import summary
        sys.stdout.reconfigure(encoding="utf-8")
        input_dim = 30
        dendritic = DendriticNetwork(input_dim=input_dim, hidden_neurons1=32, hidden_neurons2=16, branches=4, hidden_per_branch=4)
        mlp = MLPBaseline(input_dim=input_dim, match_params=sum(p.numel() for p in dendritic.parameters()))
        print("\n=== DendriticNetwork ===")
        summary(dendritic, input_size=(1, input_dim))
        print(f"Size: {dendritic.size_bytes():,} bytes")
        print("\n=== MLPBaseline ===")
        summary(mlp, input_size=(1, input_dim))
        return

    n = len(args.exp)
    tag = "all" if n == len(ALL_EXPERIMENTS) else f"{n}exp"
    label = f"{tag}_epo{args.epochs}"
    run_dir = make_run_dir(label=label)
    _save_utils.set_fig_dir(os.path.join(run_dir, "figures"))
    print(f"\n=== Running: {', '.join(args.exp)} | epochs={args.epochs} ===")
    print(f"    Output dir: {run_dir}\n")

    _fine_tune_runners = {"adult", "har"}

    results, timings = {}, {}
    pbar = tqdm(total=len(args.exp), desc="Experiments", colour="cyan")
    for key in args.exp:
        if key in _fine_tune_runners:
            REGISTRY[key](results, timings, args.epochs, SEEDS, args.fine_tune_epochs)
        else:
            REGISTRY[key](results, timings, args.epochs, SEEDS)
        pbar.set_postfix({"done": key})
        pbar.update(1)
    pbar.close()

    print_summary(results, timings)
    generate_plots(results)
    save_metrics_csv(results, run_dir)
    save_summary_txt(results, timings, run_dir)

    return results


if __name__ == "__main__":
    import os
    main()
