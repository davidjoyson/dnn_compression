import argparse
import os
import pickle
import sys
import time

from tqdm import tqdm

from src.experiments.ablation_study import (
    run_ablation, run_compression_component_ablation, run_regularization_ablation,
)
from src.experiments.har_experiment import run_har
from src.experiments.ecg_experiment import run_ecg
from src.experiments.eeg_experiment import run_eeg
from src.experiments.hapt_experiment import run_hapt

from src.loaders.load_ecg import load_ecg
from src.loaders.load_har import load_har
from src.loaders.load_eeg import load_eeg
from src.loaders.load_hapt import load_hapt

from src.models.dendritic_network import DendriticNetwork
from src.models.mlp_baseline import MLPBaseline

import src.plots.save_utils as _save_utils

from src.reporting import (
    store_simple, make_run_dir,
    save_metrics_csv, save_per_seed_csv, save_summary_txt,
    print_summary, generate_plots,
)

# ------------------------------------------------------------------ #
# Defaults                                                            #
# ------------------------------------------------------------------ #

SEEDS  = (42, 0, 7, 1, 2, 3, 4, 5, 6, 8)
EPOCHS = 50

ALL_EXPERIMENTS = ["har", "ecg", "eeg", "hapt", "ablation", "component", "regularization"]
_DEFAULT_EXPERIMENTS = ["har", "ecg", "eeg", "hapt"]

class _Tee:
    """Mirror a stream to both the terminal and a log file."""
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh
    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
    def flush(self):
        self._stream.flush()
        self._fh.flush()
    def __getattr__(self, name):
        return getattr(self._stream, name)


# ------------------------------------------------------------------ #
# Experiment runners (each fills results + timings)                   #
# ------------------------------------------------------------------ #

def _run_ablation(results, timings, epochs, seeds):
    print("\n=== Ablation Study ===\n")
    X_tr, y_tr, X_te, y_te = load_ecg()
    configs = [
        {"h1": 16, "h2":  8, "branches": 2, "hidden_per_branch": 4},
        {"h1": 32, "h2": 16, "branches": 4, "hidden_per_branch": 4},
        {"h1": 64, "h2": 32, "branches": 6, "hidden_per_branch": 4},
    ]
    t0 = time.time()
    results["Ablation Study"] = run_ablation(
        configs=configs, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
        epochs=epochs, num_classes=5,
    )
    timings["Ablation Study"] = time.time() - t0


# Same DendriticNetwork shape used for the main per-dataset experiments
# (base_experiment.py), so ablation results are comparable to the main results.
_ABLATION_CONFIG = {"h1": 64, "h2": 32, "branches": 8, "hidden_per_branch": 8}

_ABLATION_DATASETS = {
    "har":  (lambda: load_har(),  6),
    "ecg":  (lambda: load_ecg(),  5),
    "eeg":  (lambda: load_eeg(),  3),
    "hapt": (lambda: load_hapt(), 12),
}


def _run_component(results, timings, epochs, seeds):
    print("\n=== Component Ablation ===\n")
    t0 = time.time()
    out = {}
    for name, (loader, num_classes) in _ABLATION_DATASETS.items():
        print(f"  -- dataset: {name} --")
        X_tr, y_tr, X_te, y_te = loader()
        out[name] = run_compression_component_ablation(
            X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
            config=_ABLATION_CONFIG, epochs=epochs, seeds=seeds, num_classes=num_classes,
        )
    results["Component Ablation"] = out
    timings["Component Ablation"] = time.time() - t0


def _run_regularization(results, timings, epochs, seeds):
    print("\n=== Regularization Ablation ===\n")
    t0 = time.time()
    out = {}
    for name, (loader, num_classes) in _ABLATION_DATASETS.items():
        print(f"  -- dataset: {name} --")
        X_tr, y_tr, X_te, y_te = loader()
        out[name] = run_regularization_ablation(
            X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te,
            config=_ABLATION_CONFIG, epochs=epochs, seeds=seeds, num_classes=num_classes,
        )
    results["Regularization Ablation"] = out
    timings["Regularization Ablation"] = time.time() - t0


_EXP_TABLE = {
    "har":  ("UCI HAR (Wearable Sensors)",        "UCI HAR",       run_har),
    "ecg":  ("ECG Heartbeat (MIT-BIH, 5-class)",  "ECG Heartbeat", run_ecg),
    "eeg":  ("EEG Brainwave (Emotions, 3-class)",  "EEG Brainwave", run_eeg),
    "hapt": ("HAPT (UCI Smartphone, 12-class)",    "HAPT",          run_hapt),
}

_ABLATION_REGISTRY = {
    "ablation":       _run_ablation,
    "component":      _run_component,
    "regularization": _run_regularization,
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
        "--exp", nargs="+", choices=ALL_EXPERIMENTS, default=_DEFAULT_EXPERIMENTS, metavar="EXP",
        help="Experiments to run (default: har ecg eeg hapt).\nChoices: " + ", ".join(ALL_EXPERIMENTS),
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Training epochs per experiment (default: {EPOCHS})")
    parser.add_argument("--fine-tune-epochs", type=int, default=3,
                        help="Post-quantization fine-tuning epochs for har/ecg/eeg (default: 3)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                        help=f"Random seeds to average over (default: {list(SEEDS)})")
    parser.add_argument("--arch", action="store_true", help="Print model architectures and exit")
    parser.add_argument(
        "--replot", nargs="+", metavar="RUN_DIR",
        help="Load results.pkl from one or more run dirs, merge, and regenerate plots without re-training.",
    )
    args = parser.parse_args()

    if args.arch:
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
        print(f"Size: {mlp.size_bytes():,} bytes")
        return

    if args.replot:
        merged_results, merged_timings = {}, {}
        for d in args.replot:
            pkl_path = os.path.join(d, "results.pkl")
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            merged_results.update(data["results"])
            merged_timings.update(data["timings"])
        run_dir = make_run_dir(label="replot")
        _save_utils.set_fig_dir(os.path.join(run_dir, "figures"))
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(f"\n=== Replot from {len(args.replot)} run(s) -> {run_dir} ===\n")
        generate_plots(merged_results)
        print_summary(merged_results, merged_timings)
        save_metrics_csv(merged_results, run_dir)
        save_per_seed_csv(merged_results, run_dir)
        save_summary_txt(merged_results, merged_timings, run_dir)
        print(f"  Figures -> {os.path.join(run_dir, 'figures')}")
        return merged_results

    label = "_".join(args.exp) + f"_epo{args.epochs}"
    run_dir = make_run_dir(label=label)
    _save_utils.set_fig_dir(os.path.join(run_dir, "figures"))
    _models_root = os.path.join(run_dir, "models")

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    log_path = os.path.join(run_dir, "run.log")
    _log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)

    _model_dirs = {
        "har":  os.path.join(_models_root, "har"),
        "ecg":  os.path.join(_models_root, "ecg"),
        "eeg":  os.path.join(_models_root, "eeg"),
        "hapt": os.path.join(_models_root, "hapt"),
    }

    results, timings = {}, {}
    try:
        print(f"\n=== Running: {', '.join(args.exp)} | epochs={args.epochs} ===")
        print(f"    Output dir: {run_dir}")
        print(f"    Log file  : {log_path}\n")

        pbar = tqdm(total=len(args.exp), desc="Experiments", colour="cyan")
        for key in args.exp:
            if key in _EXP_TABLE:
                banner, result_key, run_fn = _EXP_TABLE[key]
                print(f"\n=== {banner} ===\n")
                t0 = time.time()
                store_simple(results, timings, result_key,
                             run_fn(epochs=args.epochs, seeds=tuple(args.seeds),
                                    fine_tune_epochs=args.fine_tune_epochs,
                                    model_dir=_model_dirs.get(key)),
                             time.time() - t0)
            else:
                _ABLATION_REGISTRY[key](results, timings, args.epochs, tuple(args.seeds))
            pbar.set_postfix({"done": key})
            pbar.update(1)
        pbar.close()

        print_summary(results, timings)
        generate_plots(results)
        save_metrics_csv(results, run_dir)
        save_per_seed_csv(results, run_dir)
        save_summary_txt(results, timings, run_dir)
        pkl_path = os.path.join(run_dir, "results.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"results": results, "timings": timings}, f)
        print(f"  Results saved  -> {pkl_path}")
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        _log_fh.close()

    print(f"  Log saved      -> {log_path}")
    return results


if __name__ == "__main__":
    main()
