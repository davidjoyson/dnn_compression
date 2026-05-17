from .utils import to_float
from src.plots.plot_accuracy import plot_accuracy
from src.plots.plot_compression import plot_compression
from src.plots.plot_ablation import plot_ablation
from src.plots.plot_scaling import plot_scaling
from src.plots.plot_roc_pr import plot_roc_pr
from src.plots.plot_training_curves import plot_training_curves


def generate_plots(results):
    import torch
    print("\n=== Generating Plots ===\n")

    import math

    for name, r in results.items():
        if not isinstance(r, dict):
            continue
        acc_u_val = r.get("accuracy_uncompressed")
        if acc_u_val is None or isinstance(acc_u_val, list):
            continue
        if hasattr(acc_u_val, "numel") and acc_u_val.numel() != 1:
            continue

        slug = name.lower().replace(" ", "_")

        # Build full method dict — include any method that has a non-NaN accuracy
        def _acc(key, std_key=None):
            v = to_float(r.get(key, float("nan")))
            s = to_float(r.get(std_key, 0.0)) if std_key else 0.0
            return (v, s)

        methods = {"Uncompressed": _acc("accuracy_uncompressed", "std_uncompressed")}
        sf8 = _acc("accuracy_compressed", "std_compressed")
        if not math.isnan(sf8[0]):
            methods["Snowflake (int8)"] = sf8
        gl8 = _acc("accuracy_compressed_global", "std_compressed_global")
        if not math.isnan(gl8[0]):
            methods["Global int8"] = gl8
        dyn = _acc("accuracy_compressed_dynamic", "std_compressed_dynamic")
        if not math.isnan(dyn[0]):
            methods["Dynamic (int8)"] = dyn
        int4 = _acc("accuracy_compressed_int4", "std_compressed_int4")
        if not math.isnan(int4[0]):
            methods["Snowflake (int4)"] = int4
        mlp = _acc("accuracy_mlp_baseline", "std_mlp_baseline")
        if not math.isnan(mlp[0]):
            methods["MLP Baseline"] = mlp

        plot_accuracy(
            methods,
            title=f"{name} Accuracy",
            filename=f"{slug}_accuracy.png",
        )
        if "size_uncompressed" in r and not isinstance(r["size_uncompressed"], torch.Tensor):
            plot_compression(
                r["size_uncompressed"],
                r["size_compressed"],
                filename=f"{slug}_compression.png",
            )

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
                plot_training_curves(r["loss_history"], title=name, filename=f"{slug}_training_curves.png")
                print(f"  {name} training curves saved")
            except Exception as e:
                print(f"  Warning: Could not plot training curves for {name}: {e}")

    print("  Plots saved to figures/ directory\n")
