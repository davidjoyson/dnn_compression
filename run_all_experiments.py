import time
import torch
from tqdm import tqdm

# Simple experiments
from src.experiments.xor_experiment import run_xor
from src.experiments.wine_experiment import run_wine
from src.experiments.uci_adult_experiment import run_uci_adult_income
from src.experiments.folktables_experiment import run_folktables_income

# Complex experiments
from src.experiments.ablation_study import run_ablation
from src.experiments.scaling_experiment import run_scaling_experiment

# Data loaders
from src.data.load_adult import load_adult_income
from src.data.load_wine import load_wine

# Plotting functions
from src.plots.plot_accuracy import plot_accuracy
from src.plots.plot_compression import plot_compression
from src.plots.plot_xor_boundary import plot_xor_boundary
from src.plots.plot_ablation import plot_ablation
from src.plots.plot_scaling import plot_scaling
from src.plots.folktables_plots import (
    plot_folktables_accuracy,
    plot_folktables_size,
    plot_folktables_tradeoff
)


def run_all_experiments():

    print("\n=== Running All Experiments ===\n")

    results = {}

    # ---------------------------------------------------------
    # 1. SIMPLE EXPERIMENTS
    # ---------------------------------------------------------
    simple_experiments = [
        ("XOR", lambda: run_xor()),
        ("Wine", lambda: run_wine()),
        ("UCI Adult Income", lambda: run_uci_adult_income()),
        ("Folktables CA 2018", lambda: run_folktables_income("CA", 2018)),
    ]

    pbar = tqdm(total=len(simple_experiments), desc="Simple Experiments", colour="white")

    for name, fn in simple_experiments:
        start = time.time()
        out = fn()
        end = time.time()

        acc_u = out["accuracy"]["uncompressed"]
        acc_c = out["accuracy"]["compressed"]

        # Convert scalar tensors to floats
        if hasattr(acc_u, "item") and acc_u.ndim == 0:
            acc_u = acc_u.item()
        if hasattr(acc_c, "item") and acc_c.ndim == 0:
            acc_c = acc_c.item()

        results[name] = {
            "accuracy_uncompressed": acc_u,
            "accuracy_compressed": acc_c,
            "size_uncompressed": out["sizes"]["uncompressed"],
            "size_compressed": out["sizes"]["compressed"],
            "time_seconds": end - start
        }

        pbar.set_postfix({"exp": name})
        pbar.update(1)

    pbar.close()

    # ---------------------------------------------------------
    # 2. ABLATION STUDY
    # ---------------------------------------------------------
    print("\n=== Running Ablation Study ===\n")

    X_train, y_train, X_test, y_test = load_wine()

    ablation_configs = [
        {"h1": 16, "h2": 8, "branches": 2, "hidden_per_branch": 4},
        {"h1": 32, "h2": 16, "branches": 4, "hidden_per_branch": 4},
        {"h1": 64, "h2": 32, "branches": 6, "hidden_per_branch": 4},
    ]

    results["Ablation Study"] = run_ablation(
        configs=ablation_configs,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=50
    )

    # ---------------------------------------------------------
    # 3. SCALING EXPERIMENT
    # ---------------------------------------------------------
    print("\n=== Running Scaling Experiment ===\n")

    X, y = load_adult_income()
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    neurons1_list = [16, 32]
    neurons2_list = [8, 16]
    branches_list = [2, 4]

    results["Scaling Experiment"] = run_scaling_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        neurons1_list=neurons1_list,
        neurons2_list=neurons2_list,
        branches_list=branches_list,
        hidden_per_branch=4,
        epochs=50
    )

    # ---------------------------------------------------------
    # 4. SUMMARY (FULLY ROBUST)
    # ---------------------------------------------------------
    print("\n=== Experiment Summary ===\n")

    for name, r in results.items():

        # Skip lists (Scaling, Ablation)
        if isinstance(r, list):
            print(f"{name}: Completed ({len(r)} configurations)\n")
            continue

        # Skip dicts without simple metrics
        if not isinstance(r, dict) or "accuracy_uncompressed" not in r:
            print(f"{name}: Completed (complex output)\n")
            continue

        acc_u = r["accuracy_uncompressed"]
        acc_c = r["accuracy_compressed"]

        # Check if tensors are actually scalars before conversion
        if hasattr(acc_u, "numel") and acc_u.numel() != 1:
            print(f"{name}: Completed (multi-dimensional results)\n")
            continue
        if hasattr(acc_c, "numel") and acc_c.numel() != 1:
            print(f"{name}: Completed (multi-dimensional results)\n")
            continue

        # Convert tensors to floats
        if hasattr(acc_u, "item"):
            acc_u = float(acc_u.item())
        else:
            acc_u = float(acc_u)
        if hasattr(acc_c, "item"):
            acc_c = float(acc_c.item())
        else:
            acc_c = float(acc_c)

        print(f"{name}:")
        print(f"  Uncompressed Acc: {acc_u:.4f}")
        print(f"  Compressed Acc:   {acc_c:.4f}")
        print(f"  Size (U -> C):    {r['size_uncompressed']} -> {r['size_compressed']} bytes")
        print(f"  Time:             {r['time_seconds']:.2f} sec\n")

    # ---------------------------------------------------------
    # 5. PLOTTING
    # ---------------------------------------------------------
    print("\n=== Generating Plots ===\n")

    # Plot individual experiment results
    for name, r in results.items():
        if isinstance(r, dict) and "accuracy_uncompressed" in r and "accuracy_compressed" in r:
            if hasattr(r["accuracy_uncompressed"], "numel"):
                if r["accuracy_uncompressed"].numel() == 1:
                    acc_u = float(r["accuracy_uncompressed"].item()) if hasattr(r["accuracy_uncompressed"], "item") else float(r["accuracy_uncompressed"])
                    acc_c = float(r["accuracy_compressed"].item()) if hasattr(r["accuracy_compressed"], "item") else float(r["accuracy_compressed"])
                    
                    # Plot accuracy
                    plot_accuracy(acc_u, acc_c, title=f"{name} Accuracy", filename=f"{name.lower().replace(' ', '_')}_accuracy.png")
                    
                    # Plot compression ratio
                    if "size_uncompressed" in r and "size_compressed" in r:
                        plot_compression(r["size_uncompressed"], r["size_compressed"], 
                                       filename=f"{name.lower().replace(' ', '_')}_compression.png")
            else:
                if isinstance(r["accuracy_uncompressed"], (int, float)):
                    plot_accuracy(r["accuracy_uncompressed"], r["accuracy_compressed"], 
                                 title=f"{name} Accuracy", filename=f"{name.lower().replace(' ', '_')}_accuracy.png")
                    if "size_uncompressed" in r:
                        plot_compression(r["size_uncompressed"], r["size_compressed"],
                                       filename=f"{name.lower().replace(' ', '_')}_compression.png")
    
    # Plot ablation study results if available
    if "Ablation Study" in results and isinstance(results["Ablation Study"], list):
        try:
            ablation_data = results["Ablation Study"]
            # Convert ablation results to dict format for plotting
            ablation_dict = {}
            for i, res in enumerate(ablation_data):
                config_str = f"Config {i+1}"
                ablation_dict[config_str] = float(res["accuracy_uncompressed"].item() 
                                                  if hasattr(res["accuracy_uncompressed"], "item") 
                                                  else res["accuracy_uncompressed"])
            if ablation_dict:
                plot_ablation(ablation_dict, filename="ablation_study.png")
                print("  Ablation plot saved")
        except Exception as e:
            print(f"  Warning: Could not plot ablation study: {e}")
    
    # Plot scaling experiment results if available
    if "Scaling Experiment" in results:
        try:
            scaling_results = results["Scaling Experiment"]
            if isinstance(scaling_results, dict):
                print("  Scaling experiment plots skipped (multi-dimensional data)")
        except Exception as e:
            print(f"  Warning: Could not process scaling experiment: {e}")

    print("  Plots saved to figures/ directory\n")

    return results


if __name__ == "__main__":
    run_all_experiments()
