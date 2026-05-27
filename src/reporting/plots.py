from .utils import to_float
from src.plots.plot_accuracy import plot_accuracy
from src.plots.plot_compression import plot_compression
from src.plots.plot_ablation import plot_ablation
from src.plots.plot_roc_pr import plot_roc_pr
from src.plots.plot_training_curves import plot_training_curves
from src.plots.plot_confusion_matrix import plot_confusion_matrix
from src.plots.plot_cross_dataset import plot_cross_dataset_summary
from src.plots.plot_pareto import plot_pareto
from src.plots.plot_per_class_f1 import plot_per_class_f1
from src.plots.plot_weight_dist import plot_weight_distribution
from src.plots.plot_inference_time import plot_inference_time
from src.plots.plot_val_accuracy import plot_val_accuracy


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
        mlp = _acc("accuracy_mlp_baseline", "std_mlp_baseline")
        if not math.isnan(mlp[0]):
            methods["MLP Baseline"] = mlp

        plot_accuracy(
            methods,
            title=f"{name} Accuracy",
            filename=f"{slug}_accuracy.png",
        )

        # F1 plot — same method structure, different values
        f1_methods = {"Uncompressed": _acc("f1_uncompressed", "std_f1_uncompressed")}
        f1_sf8 = _acc("f1_compressed", "std_f1_compressed")
        if not math.isnan(f1_sf8[0]):
            f1_methods["Snowflake (int8)"] = f1_sf8
        f1_gl8 = _acc("f1_compressed_global", "std_f1_compressed_global")
        if not math.isnan(f1_gl8[0]):
            f1_methods["Global int8"] = f1_gl8
        f1_dyn = _acc("f1_compressed_dynamic", "std_f1_compressed_dynamic")
        if not math.isnan(f1_dyn[0]):
            f1_methods["Dynamic (int8)"] = f1_dyn
        f1_mlp = _acc("f1_mlp_baseline", "std_f1_mlp_baseline")
        if not math.isnan(f1_mlp[0]):
            f1_methods["MLP Baseline"] = f1_mlp
        if not math.isnan(f1_methods["Uncompressed"][0]):
            plot_accuracy(
                f1_methods,
                title=f"{name} F1 Score",
                filename=f"{slug}_f1.png",
                ylabel="Macro F1" if len(f1_methods) > 1 else "F1",
            )

        if "size_uncompressed" in r and not isinstance(r["size_uncompressed"], torch.Tensor):
            import math as _math
            _sizes = {"Uncompressed": r["size_uncompressed"]}
            _sf8 = r.get("size_compressed")
            if _sf8 is not None and not (isinstance(_sf8, float) and _math.isnan(_sf8)):
                _sizes["Snowflake (int8)"] = _sf8
            _gl8 = r.get("size_compressed_global")
            if _gl8:
                _sizes["Global int8"] = _gl8
            _dyn = r.get("size_compressed_dynamic")
            if _dyn:
                _sizes["Dynamic (int8)"] = _dyn
            plot_compression(
                _sizes,
                title=f"{name} Model Size",
                filename=f"{slug}_compression.png",
            )

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

    for name, r in results.items():
        if isinstance(r, dict) and r.get("conf_matrix") is not None:
            try:
                slug = name.lower().replace(" ", "_")
                plot_confusion_matrix(
                    r["conf_matrix"],
                    title=name,
                    filename=f"{slug}_confusion.png",
                    class_names=r.get("class_names"),
                )
                print(f"  {name} confusion matrix saved")
            except Exception as e:
                print(f"  Warning: Could not plot confusion matrix for {name}: {e}")

    # Cross-dataset summary + Pareto (need >=2 datasets with accuracy data)
    ds_results = {
        name: r for name, r in results.items()
        if isinstance(r, dict) and "accuracy_uncompressed" in r
        and not isinstance(r.get("accuracy_uncompressed"), list)
    }
    if len(ds_results) >= 2:
        try:
            plot_cross_dataset_summary(ds_results)
            print("  Cross-dataset summary saved")
        except Exception as e:
            print(f"  Warning: Could not plot cross-dataset summary: {e}")
        try:
            plot_pareto(ds_results)
            print("  Pareto plot saved")
        except Exception as e:
            print(f"  Warning: Could not plot pareto: {e}")
        try:
            plot_inference_time(ds_results)
            print("  Inference time plot saved")
        except Exception as e:
            print(f"  Warning: Could not plot inference time: {e}")

    # Per-dataset plots requiring new data
    for name, r in results.items():
        if not isinstance(r, dict):
            continue
        slug = name.lower().replace(" ", "_")

        if r.get("conf_matrix") is not None and r.get("class_names") is not None:
            try:
                plot_per_class_f1(r["conf_matrix"], r["class_names"],
                                  title=name, filename=f"{slug}_per_class_f1.png")
                print(f"  {name} per-class F1 saved")
            except Exception as e:
                print(f"  Warning: Could not plot per-class F1 for {name}: {e}")

        if r.get("weight_dist") is not None:
            try:
                plot_weight_distribution(r["weight_dist"], title=name,
                                         filename=f"{slug}_weight_dist.png")
                print(f"  {name} weight distribution saved")
            except Exception as e:
                print(f"  Warning: Could not plot weight dist for {name}: {e}")

        if r.get("val_acc_history") is not None:
            try:
                plot_val_accuracy(r["val_acc_history"], title=name,
                                  filename=f"{slug}_val_accuracy.png")
                print(f"  {name} val accuracy curve saved")
            except Exception as e:
                print(f"  Warning: Could not plot val accuracy for {name}: {e}")

    print("  Plots saved to figures/ directory\n")
