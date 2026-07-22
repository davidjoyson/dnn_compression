from src.plots.plot_compression import plot_compression_by_dataset
from src.plots.plot_ablation import plot_ablation
from src.plots.plot_roc_pr import plot_roc_pr
from src.plots.plot_training_curves import plot_training_curves
from src.plots.plot_confusion_matrix import plot_confusion_matrix
from src.plots.plot_cross_dataset import plot_cross_dataset_summary, plot_cross_dataset_f1
from src.plots.plot_pareto import plot_pareto
from src.plots.plot_per_class_f1 import plot_per_class_f1
from src.plots.plot_weight_dist import plot_weight_distribution
from src.plots.plot_inference_time import plot_inference_time
from src.plots.plot_val_accuracy import plot_val_accuracy
from src.plots.plot_component_ablation import plot_component_ablation, plot_ablation_combined
from src.plots.plot_compression_delta import plot_compression_delta
from src.plots.plot_edge_profile import plot_edge_profile
from src.plots.plot_branch_diversity import plot_branch_diversity


def generate_plots(results):
    print("\n=== Generating Plots ===\n")

    # Accuracy, F1, and model size are combined into one cross-dataset chart
    # each (see the ds_results block below) instead of one file per dataset.

    if "Component Ablation" in results and isinstance(results["Component Ablation"], dict):
        for dataset, conditions in results["Component Ablation"].items():
            try:
                plot_component_ablation(conditions,
                                        filename=f"component_ablation_{dataset}.png",
                                        title=f"Compression Component Ablation ({dataset})")
                print(f"  Component ablation plot saved ({dataset})")
            except Exception as e:
                print(f"  Warning: Could not plot component ablation ({dataset}): {e}")
        try:
            plot_ablation_combined(results["Component Ablation"],
                                   filename="combined/component_ablation_all.png",
                                   title="Compression Component Ablation (all datasets)")
            print("  Component ablation combined plot saved")
        except Exception as e:
            print(f"  Warning: Could not plot combined component ablation: {e}")

    if "Regularization Ablation" in results and isinstance(results["Regularization Ablation"], dict):
        for dataset, conditions in results["Regularization Ablation"].items():
            try:
                plot_component_ablation(conditions,
                                        filename=f"regularization_ablation_{dataset}.png",
                                        title=f"Regularization vs. Quantization ({dataset})")
                print(f"  Regularization ablation plot saved ({dataset})")
            except Exception as e:
                print(f"  Warning: Could not plot regularization ablation ({dataset}): {e}")
        try:
            plot_ablation_combined(results["Regularization Ablation"],
                                   filename="combined/regularization_ablation_all.png",
                                   title="Regularization vs. Quantization (all datasets)")
            print("  Regularization ablation combined plot saved")
        except Exception as e:
            print(f"  Warning: Could not plot combined regularization ablation: {e}")

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

    if "Ablation Study" in results and isinstance(results["Ablation Study"], dict):
        try:
            arch_ablation, cond_labels = {}, {}
            for dataset, configs in results["Ablation Study"].items():
                per_cond = {}
                for i, res in enumerate(configs):
                    key = f"cfg{i}"
                    acc = res["accuracy_uncompressed"]
                    per_cond[key] = {"mean": acc["mean"], "std": acc.get("std", 0.0)}
                    cond_labels[key] = f"h1={res['config']['h1']}"
                arch_ablation[dataset] = per_cond
            if arch_ablation:
                cond_order = sorted(cond_labels, key=lambda k: int(k[3:]))
                plot_ablation_combined(arch_ablation, filename="combined/ablation_study_all.png",
                                       title="Architecture Size Ablation (all datasets)",
                                       condition_order=cond_order, condition_labels=cond_labels)
                print("  Architecture ablation combined plot saved")
        except Exception as e:
            print(f"  Warning: Could not plot combined architecture ablation: {e}")

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
            plot_cross_dataset_f1(ds_results)
            print("  Cross-dataset F1 saved")
        except Exception as e:
            print(f"  Warning: Could not plot cross-dataset F1: {e}")
        try:
            plot_compression_by_dataset(ds_results)
            print("  Cross-dataset model size saved")
        except Exception as e:
            print(f"  Warning: Could not plot cross-dataset model size: {e}")
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
        try:
            plot_edge_profile(ds_results)
            print("  Edge profile plot saved")
        except Exception as e:
            print(f"  Warning: Could not plot edge profile: {e}")

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
            try:
                plot_compression_delta(r["conf_matrix"], r["class_names"],
                                       title=name, filename=f"{slug}_compression_delta.png")
                print(f"  {name} compression delta saved")
            except Exception as e:
                print(f"  Warning: Could not plot compression delta for {name}: {e}")

        if r.get("branch_diversity") is not None:
            try:
                plot_branch_diversity(r["branch_diversity"], title=name,
                                      filename=f"{slug}_branch_diversity.png",
                                      control_spread=r.get("branch_diversity_control"))
                print(f"  {name} branch diversity saved")
            except Exception as e:
                print(f"  Warning: Could not plot branch diversity for {name}: {e}")

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
