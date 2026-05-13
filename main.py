import argparse
import csv
import io
import math
import os
import time
from datetime import datetime

import torch
from tqdm import tqdm

from src.experiments.wine_experiment import run_wine
from src.experiments.uci_adult_experiment import run_uci_adult_income
from src.experiments.folktables_experiment import run_folktables_income
from src.experiments.ablation_study import run_ablation, run_compression_component_ablation
from src.experiments.scaling_experiment import run_scaling_experiment
from src.experiments.folktables_multistate_experiment import run_folktables_multistate
from src.experiments.creditcard_experiment import run_creditcard_fraud

from src.data.load_adult import load_adult_income
from src.data.load_wine import load_wine

from src.models.dendritic_network import DendriticNetwork
from src.models.mlp_baseline import MLPBaseline

import src.plots.save_utils as _save_utils

from src.plots.plot_accuracy import plot_accuracy
from src.plots.plot_compression import plot_compression
from src.plots.plot_ablation import plot_ablation
from src.plots.plot_scaling import plot_scaling
from src.plots.plot_folktables_multistate import plot_folktables_multistate
from src.plots.plot_roc_pr import plot_roc_pr
from src.plots.plot_training_curves import plot_training_curves

# ------------------------------------------------------------------ #
# Defaults                                                            #
# ------------------------------------------------------------------ #

SEEDS  = (42, 0, 7)
EPOCHS = 50

ALL_EXPERIMENTS = [
    "wine", "adult", "folktables",
    "ablation", "component", "scaling", "multistate", "fraud",
]

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _to_float(val):
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


def _store_simple(results, timings, name, out, elapsed):
    results[name] = {
        "accuracy_uncompressed": _to_float(out["accuracy"]["uncompressed"]),
        "accuracy_compressed":   _to_float(out["accuracy"]["compressed"]),
        "accuracy_mlp_baseline": _to_float(out["accuracy"].get("mlp_baseline", float("nan"))),
        "std_uncompressed": _to_float(out.get("accuracy_std", {}).get("uncompressed", 0.0)),
        "std_compressed":   _to_float(out.get("accuracy_std", {}).get("compressed",   0.0)),
        "std_mlp_baseline": _to_float(out.get("accuracy_std", {}).get("mlp_baseline", 0.0)),
        "size_uncompressed":     out["sizes"]["uncompressed"],
        "size_compressed":       out["sizes"]["compressed"],
        "size_mlp_uncompressed": out["sizes"].get("mlp_uncompressed"),
        "size_mlp_compressed":   out["sizes"].get("mlp_compressed"),
        "time_seconds": elapsed,
        "mse_uncompressed":     _to_float(out.get("mse", {}).get("uncompressed", float("nan"))),
        "mse_compressed":       _to_float(out.get("mse", {}).get("compressed",   float("nan"))),
        "mse_mlp_baseline":     _to_float(out.get("mse", {}).get("mlp_baseline", float("nan"))),
        "mse_std_uncompressed":   _to_float(out.get("mse_std", {}).get("uncompressed",   0.0)),
        "mse_std_compressed":     _to_float(out.get("mse_std", {}).get("compressed",     0.0)),
        "mse_std_mlp_baseline":   _to_float(out.get("mse_std", {}).get("mlp_baseline",   0.0)),
        "mse_std_mlp_compressed": _to_float(out.get("mse_std", {}).get("mlp_compressed", 0.0)),
        "accuracy_mlp_compressed": _to_float(out["accuracy"].get("mlp_compressed", float("nan"))),
        "std_mlp_compressed":      _to_float(out.get("accuracy_std", {}).get("mlp_compressed", 0.0)),
        "mse_mlp_compressed":      _to_float(out.get("mse", {}).get("mlp_compressed", float("nan"))),
        "num_seeds": out.get("num_seeds", 1),
        "curve_data": out.get("curve_data"),
        "loss_history": out.get("loss_history"),
    }
    timings[name] = elapsed


# ------------------------------------------------------------------ #
# Run directory + output helpers                                      #
# ------------------------------------------------------------------ #

def _make_run_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", f"run_{stamp}")
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    return run_dir


def _save_metrics_csv(results, run_dir):
    rows = []
    for name, r in results.items():
        if not isinstance(r, dict) or "accuracy_uncompressed" not in r:
            continue
        _acc_check = r["accuracy_uncompressed"]
        if isinstance(_acc_check, list) or (hasattr(_acc_check, "numel") and _acc_check.numel() > 1):
            continue  # multi-value result (Scaling Experiment, Folktables Multi-State) — not a flat row
        acc_u   = _to_float(r.get("accuracy_uncompressed",   float("nan")))
        acc_c   = _to_float(r.get("accuracy_compressed",     float("nan")))
        acc_mlp = _to_float(r.get("accuracy_mlp_baseline",   float("nan")))
        acc_mlp_c = _to_float(r.get("accuracy_mlp_compressed", float("nan")))

        def _delta(a, b):
            return round(a - b, 6) if not (math.isnan(a) or math.isnan(b)) else ""

        def _ratio(u, c):
            return round(u / c, 3) if c and not math.isnan(u) else ""

        rows.append({
            "experiment":              name,
            "acc_uncompressed":        round(acc_u,   6),
            "acc_compressed":          round(acc_c,   6),
            "acc_mlp_baseline":        round(acc_mlp, 6) if not math.isnan(acc_mlp) else "",
            "acc_mlp_compressed":      round(acc_mlp_c, 6) if not math.isnan(acc_mlp_c) else "",
            "acc_delta_dendritic":     _delta(acc_c,     acc_u),
            "acc_delta_mlp":           _delta(acc_mlp_c, acc_mlp),
            "std_uncompressed":        r.get("std_uncompressed",   0.0),
            "std_compressed":          r.get("std_compressed",     0.0),
            "std_mlp_baseline":        r.get("std_mlp_baseline",   0.0),
            "std_mlp_compressed":      r.get("std_mlp_compressed", 0.0),
            "mse_uncompressed":        r.get("mse_uncompressed",   ""),
            "mse_compressed":          r.get("mse_compressed",     ""),
            "mse_mlp_baseline":        r.get("mse_mlp_baseline",   ""),
            "mse_mlp_compressed":      r.get("mse_mlp_compressed", ""),
            "size_dendritic_u":        r.get("size_uncompressed",     ""),
            "size_dendritic_c":        r.get("size_compressed",       ""),
            "compression_ratio_dendritic": _ratio(r.get("size_uncompressed", float("nan")),
                                                   r.get("size_compressed",   float("nan"))),
            "size_mlp_u":              r.get("size_mlp_uncompressed", ""),
            "size_mlp_c":              r.get("size_mlp_compressed",   ""),
            "compression_ratio_mlp":   _ratio(r.get("size_mlp_uncompressed", float("nan")),
                                               r.get("size_mlp_compressed",   float("nan"))),
            "time_seconds":            round(r.get("time_seconds", 0.0), 2),
            "num_seeds":               r.get("num_seeds", 1),
        })

    if not rows:
        return
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Metrics saved -> {csv_path}")


def _save_summary_txt(results, timings, run_dir):
    buf = io.StringIO()
    import sys
    old_stdout = sys.stdout
    sys.stdout = buf
    _print_summary(results, timings)
    sys.stdout = old_stdout
    txt_path = os.path.join(run_dir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  Summary saved  -> {txt_path}")


# ------------------------------------------------------------------ #
# Experiment runners (each fills results + timings)                   #
# ------------------------------------------------------------------ #

def _run_wine(results, timings, epochs, seeds):
    print("\n=== Wine ===\n")
    t0  = time.time()
    out = run_wine(epochs=epochs, seeds=seeds)
    _store_simple(results, timings, "Wine", out, time.time() - t0)


def _run_adult(results, timings, epochs, seeds):
    print("\n=== UCI Adult Income ===\n")
    t0  = time.time()
    out = run_uci_adult_income(epochs=epochs, seeds=seeds)
    _store_simple(results, timings, "UCI Adult Income", out, time.time() - t0)


def _run_folktables(results, timings, epochs, seeds):
    print("\n=== Folktables CA 2018 ===\n")
    t0  = time.time()
    out = run_folktables_income("CA", 2018, epochs=epochs)
    _store_simple(results, timings, "Folktables CA 2018", out, time.time() - t0)


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
        configs=configs,
        X_train=X_tr, y_train=y_tr,
        X_test=X_te,  y_test=y_te,
        epochs=epochs,
    )
    timings["Ablation Study"] = time.time() - t0


def _run_component(results, timings, epochs, seeds):
    print("\n=== Component Ablation ===\n")
    X_tr, y_tr, X_te, y_te = load_wine()
    config = {"h1": 32, "h2": 16, "branches": 4, "hidden_per_branch": 4}
    t0 = time.time()
    results["Component Ablation"] = run_compression_component_ablation(
        X_train=X_tr, y_train=y_tr,
        X_test=X_te,  y_test=y_te,
        config=config,
        epochs=epochs,
        seeds=seeds,
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
        X_train=X[:split],  y_train=y[:split],
        X_test=X[split:],   y_test=y[split:],
        neurons1_list=[16, 32],
        neurons2_list=[8, 16],
        branches_list=[2, 4],
        hidden_per_branch=4,
        epochs=epochs,
    )
    timings["Scaling Experiment"] = time.time() - t0


def _run_multistate(results, timings, epochs, seeds):
    print("\n=== Folktables Multi-State ===\n")
    t0 = time.time()
    results["Folktables Multi-State"] = run_folktables_multistate(
        train_state="CA",
        test_states=("CA", "TX", "NY", "FL", "WA"),
        year=2018,
        epochs=epochs,
    )
    timings["Folktables Multi-State"] = time.time() - t0


def _run_fraud(results, timings, epochs, seeds):
    print("\n=== Credit Card Fraud ===\n")
    t0  = time.time()
    out = run_creditcard_fraud(epochs=epochs, seeds=seeds)
    _store_simple(results, timings, "Credit Card Fraud", out, time.time() - t0)


REGISTRY = {
    "wine":       _run_wine,
    "adult":      _run_adult,
    "folktables": _run_folktables,
    "ablation":   _run_ablation,
    "component":  _run_component,
    "scaling":    _run_scaling,
    "multistate": _run_multistate,
    "fraud":      _run_fraud,
}

# ------------------------------------------------------------------ #
# Summary                                                             #
# ------------------------------------------------------------------ #

def _print_summary(results, timings):
    print("\n=== Experiment Summary ===\n")

    for name, r in results.items():
        _t = (r.get("time_seconds") if isinstance(r, dict) else None) or timings.get(name)
        time_str = f"  Time             : {_t:.2f} sec\n" if _t is not None else "\n"

        if isinstance(r, list):
            print(f"{name} ({len(r)} configurations):")
            for i, cfg_r in enumerate(r):
                cfg = cfg_r.get("config", {})
                au  = cfg_r.get("accuracy_uncompressed", float("nan"))
                ac  = cfg_r.get("accuracy_compressed",   float("nan"))
                mu  = cfg_r.get("mse_uncompressed",      float("nan"))
                mc  = cfg_r.get("mse_compressed",        float("nan"))
                tag = f"h1={cfg.get('h1','?')} h2={cfg.get('h2','?')} br={cfg.get('branches','?')}"
                print(f"  Config {i+1} {tag}: acc_u={au:.4f} acc_c={ac:.4f}  mse_u={mu:.4f} mse_c={mc:.4f}")
            print(time_str, end="")
            continue

        if name == "Component Ablation" and isinstance(r, dict) and "none" in r:
            print(f"{name}:")
            for condition, stats in r.items():
                mse_m = stats.get("mse_mean", float("nan"))
                mse_s = stats.get("mse_std",  0.0)
                print(f"  {condition:<12}: acc={stats['mean']:.4f} +/- {stats['std']:.4f}  mse={mse_m:.4f} +/- {mse_s:.4f}")
            print(time_str, end="")
            continue

        if name == "Folktables Multi-State" and isinstance(r, dict) and "test_states" in r:
            print(f"{name} (trained on {r['train_state']}):")
            mse_u_list = r.get("mse_uncompressed", [float("nan")] * len(r["test_states"]))
            mse_c_list = r.get("mse_compressed",   [float("nan")] * len(r["test_states"]))
            for state, au, ac, mu, mc in zip(
                r["test_states"], r["accuracy_uncompressed"], r["accuracy_compressed"],
                mse_u_list, mse_c_list,
            ):
                print(f"  {state}: u_acc={au:.4f} u_mse={mu:.4f}  c_acc={ac:.4f} c_mse={mc:.4f}")
            print(f"  Size (U -> C)    : {r['size_uncompressed']} -> {r['size_compressed']} bytes")
            print(time_str, end="")
            continue

        if not isinstance(r, dict) or "accuracy_uncompressed" not in r:
            print(f"{name}: Completed (complex output)")
            print(time_str, end="")
            continue

        acc_u = r["accuracy_uncompressed"]
        if hasattr(acc_u, "numel") and acc_u.numel() != 1:
            print(f"{name}: Completed (multi-dimensional results)")
            print(time_str, end="")
            continue

        acc_u   = _to_float(acc_u)
        acc_c   = _to_float(r["accuracy_compressed"])
        acc_mlp = _to_float(r.get("accuracy_mlp_baseline", float("nan")))
        std_u   = _to_float(r.get("std_uncompressed", 0.0))
        std_c   = _to_float(r.get("std_compressed",   0.0))
        std_mlp = _to_float(r.get("std_mlp_baseline", 0.0))

        n_seeds   = r.get("num_seeds", 1)
        seed_note = f" (mean over {n_seeds} seeds)" if n_seeds > 1 else ""

        mse_u   = _to_float(r.get("mse_uncompressed",     float("nan")))
        mse_c   = _to_float(r.get("mse_compressed",       float("nan")))
        mse_mlp = _to_float(r.get("mse_mlp_baseline",     float("nan")))
        std_mse_u   = _to_float(r.get("mse_std_uncompressed", 0.0))
        std_mse_c   = _to_float(r.get("mse_std_compressed",   0.0))
        std_mse_mlp = _to_float(r.get("mse_std_mlp_baseline", 0.0))

        print(f"{name}{seed_note}:")
        print(f"  Uncompressed Acc : {acc_u:.4f} +/- {std_u:.4f}")
        print(f"  Compressed Acc   : {acc_c:.4f} +/- {std_c:.4f}")
        acc_mlp_c   = _to_float(r.get("accuracy_mlp_compressed", float("nan")))
        std_mlp_c   = _to_float(r.get("std_mlp_compressed",      0.0))
        mse_mlp_c   = _to_float(r.get("mse_mlp_compressed",      float("nan")))
        std_mse_mlp_c = _to_float(r.get("mse_std_mlp_compressed", 0.0))

        if acc_mlp == acc_mlp:
            print(f"  MLP Baseline Acc : {acc_mlp:.4f} +/- {std_mlp:.4f}")
        if acc_mlp_c == acc_mlp_c:
            delta_d   = acc_c   - acc_u
            delta_mlp = acc_mlp_c - acc_mlp
            print(f"  MLP Compressed   : {acc_mlp_c:.4f} +/- {std_mlp_c:.4f}")
            print(f"  -- Acc delta (Dendritic): {delta_d:+.4f}  |  MLP: {delta_mlp:+.4f}")
        if mse_u == mse_u:
            print(f"  Uncompressed MSE : {mse_u:.4f} +/- {std_mse_u:.4f}")
            print(f"  Compressed MSE   : {mse_c:.4f} +/- {std_mse_c:.4f}")
        if mse_mlp == mse_mlp:
            print(f"  MLP Baseline MSE : {mse_mlp:.4f} +/- {std_mse_mlp:.4f}")
        if mse_mlp_c == mse_mlp_c:
            print(f"  MLP Compressed MSE:{mse_mlp_c:.4f} +/- {std_mse_mlp_c:.4f}")
        print(f"  Dendritic Size   : {r['size_uncompressed']} -> {r['size_compressed']} bytes")
        if r.get("size_mlp_uncompressed") is not None:
            print(f"  MLP Size         : {r['size_mlp_uncompressed']} -> {r['size_mlp_compressed']} bytes")
        print(time_str, end="")


# ------------------------------------------------------------------ #
# Plotting                                                            #
# ------------------------------------------------------------------ #

def _generate_plots(results):
    print("\n=== Generating Plots ===\n")

    for name, r in results.items():
        if not isinstance(r, dict):
            continue
        acc_u_val = r.get("accuracy_uncompressed")
        if acc_u_val is None:
            continue
        if isinstance(acc_u_val, list):
            continue
        if hasattr(acc_u_val, "numel") and acc_u_val.numel() != 1:
            continue
        plot_accuracy(
            _to_float(acc_u_val),
            _to_float(r["accuracy_compressed"]),
            title=f"{name} Accuracy",
            filename=f"{name.lower().replace(' ', '_')}_accuracy.png",
        )
        if "size_uncompressed" in r and not isinstance(r["size_uncompressed"], torch.Tensor):
            plot_compression(
                r["size_uncompressed"],
                r["size_compressed"],
                filename=f"{name.lower().replace(' ', '_')}_compression.png",
            )

    if "Folktables Multi-State" in results:
        try:
            plot_folktables_multistate(results["Folktables Multi-State"])
            print("  Folktables multi-state plot saved")
        except Exception as e:
            print(f"  Warning: Could not plot multi-state: {e}")

    if "Scaling Experiment" in results:
        try:
            plot_scaling(
                results["Scaling Experiment"],
                neurons1_list=[16, 32],
                neurons2_list=[8, 16],
                branches_list=[2, 4],
            )
            print("  Scaling plots saved")
        except Exception as e:
            print(f"  Warning: Could not plot scaling: {e}")

    if "Ablation Study" in results and isinstance(results["Ablation Study"], list):
        try:
            ablation_dict = {
                f"Config {i+1}": float(
                    res["accuracy_uncompressed"].item()
                    if hasattr(res["accuracy_uncompressed"], "item")
                    else res["accuracy_uncompressed"]
                )
                for i, res in enumerate(results["Ablation Study"])
            }
            if ablation_dict:
                plot_ablation(ablation_dict, filename="ablation_study.png")
                print("  Ablation plot saved")
        except Exception as e:
            print(f"  Warning: Could not plot ablation: {e}")

    for name, r in results.items():
        if isinstance(r, dict) and r.get("curve_data") is not None:
            try:
                slug = name.lower().replace(" ", "_")
                plot_roc_pr(r["curve_data"], title=name, filename=f"{slug}_roc_pr.png")
                print(f"  {name} ROC/PR curves saved")
            except Exception as e:
                print(f"  Warning: Could not plot ROC/PR for {name}: {e}")

    for name, r in results.items():
        if isinstance(r, dict) and r.get("loss_history") is not None:
            try:
                slug = name.lower().replace(" ", "_")
                plot_training_curves(
                    r["loss_history"],
                    title=name,
                    filename=f"{slug}_training_curves.png",
                )
                print(f"  {name} training curves saved")
            except Exception as e:
                print(f"  Warning: Could not plot training curves for {name}: {e}")

    print("  Plots saved to figures/ directory\n")


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="DNN Compression Experiments",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        choices=ALL_EXPERIMENTS,
        default=ALL_EXPERIMENTS,
        metavar="EXP",
        help=(
            "Experiments to run (default: all).\n"
            "Choices: " + ", ".join(ALL_EXPERIMENTS)
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Training epochs per experiment (default: {EPOCHS})",
    )
    parser.add_argument(
        "--arch",
        action="store_true",
        help="Print model architectures and exit",
    )
    args = parser.parse_args()

    if args.arch:
        import sys
        from torchinfo import summary
        sys.stdout.reconfigure(encoding="utf-8")
        input_dim = 30
        dendritic = DendriticNetwork(input_dim=input_dim, hidden_neurons1=32, hidden_neurons2=16, branches=4, hidden_per_branch=4)
        mlp       = MLPBaseline(input_dim=input_dim, hidden=32)
        print("\n=== DendriticNetwork ===")
        summary(dendritic, input_size=(1, input_dim))
        print(f"Size: {dendritic.size_bytes():,} bytes")
        print("\n=== MLPBaseline ===")
        summary(mlp, input_size=(1, input_dim))
        return

    run_dir = _make_run_dir()
    _save_utils.set_fig_dir(os.path.join(run_dir, "figures"))
    print(f"\n=== Running: {', '.join(args.exp)} | epochs={args.epochs} ===")
    print(f"    Output dir: {run_dir}\n")

    results = {}
    timings = {}

    pbar = tqdm(total=len(args.exp), desc="Experiments", colour="cyan")
    for key in args.exp:
        REGISTRY[key](results, timings, args.epochs, SEEDS)
        pbar.set_postfix({"done": key})
        pbar.update(1)
    pbar.close()

    _print_summary(results, timings)
    _generate_plots(results)
    _save_metrics_csv(results, run_dir)
    _save_summary_txt(results, timings, run_dir)

    return results


if __name__ == "__main__":
    main()
