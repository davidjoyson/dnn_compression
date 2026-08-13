"""Separate Snowflake weight-precision comparison: INT8 vs INT6 vs INT4."""

import copy
import time

import torch
from sklearn.model_selection import train_test_split

from src.analysis.tost import ci_95, tost_paired
from src.compression.compression_pipeline import (
    compress_model,
    compressed_size_bytes,
    decompress_model,
    compress_model_int6,
    decompress_model_int6,
    int6_size_bytes,
    compress_model_int4,
    decompress_model_int4,
    int4_size_bytes,
)
from src.loaders.load_ecg_patient_split import load_ecg_patient_split
from src.loaders.load_hapt import load_hapt
from src.models.dendritic_network import DendriticNetwork
from src.training.evaluate import evaluate, f1_eval
from src.training.train import train


SEEDS = (42, 0, 7)
EPOCHS = 50
FINE_TUNE_EPOCHS = 3
DATASETS = (
    ("ECG", lambda: load_ecg_patient_split(balance=True), 5),
    ("HAPT", lambda: load_hapt(balance=False), 12),
)


def _quantized_metrics(model, original_state, compressor, decompressor,
                       X_train, y_train, X_test, y_test, num_classes):
    model.load_state_dict(original_state)
    compressed = compressor(
        model,
        fine_tune_data=(X_train, y_train),
        fine_tune_epochs=FINE_TUNE_EPOCHS,
    )
    decompressor(compressed, model)
    return (
        evaluate(model, X_test, y_test, num_classes=num_classes),
        f1_eval(model, X_test, y_test, num_classes=num_classes),
        compressed,
    )


def run_precision_comparison(name, loader, num_classes):
    metrics = {
        "float32": {"accuracy": [], "f1": []},
        "int8": {"accuracy": [], "f1": []},
        "int6": {"accuracy": [], "f1": []},
        "int4": {"accuracy": [], "f1": []},
    }
    compressed_by_bits = {}
    started = time.time()

    for seed in SEEDS:
        X_train_np, y_train_np, X_test_np, y_test_np = loader()
        X_train_np, _, y_train_np, _ = train_test_split(
            X_train_np, y_train_np, test_size=0.1,
            random_state=seed, stratify=y_train_np,
        )
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.long)
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        y_test = torch.tensor(y_test_np, dtype=torch.long)

        torch.manual_seed(seed)
        model = DendriticNetwork(
            input_dim=X_train.shape[1], hidden_neurons1=64,
            hidden_neurons2=32, branches=8, hidden_per_branch=8,
            num_classes=num_classes,
        )
        train(model, X_train, y_train, epochs=EPOCHS,
              num_classes=num_classes, verbose=True,
              label=f"{name} precision seed={seed}")
        original_state = copy.deepcopy(model.state_dict())
        metrics["float32"]["accuracy"].append(
            evaluate(model, X_test, y_test, num_classes=num_classes)
        )
        metrics["float32"]["f1"].append(
            f1_eval(model, X_test, y_test, num_classes=num_classes)
        )

        methods = (
            ("int8", compress_model, decompress_model),
            ("int6", compress_model_int6, decompress_model_int6),
            ("int4", compress_model_int4, decompress_model_int4),
        )
        for label, compressor, decompressor in methods:
            acc, f1, compressed = _quantized_metrics(
                model, original_state, compressor, decompressor,
                X_train, y_train, X_test, y_test, num_classes,
            )
            metrics[label]["accuracy"].append(acc)
            metrics[label]["f1"].append(f1)
            compressed_by_bits[label] = compressed

        print(f"  {name} seed={seed}: " + "  ".join(
            f"{label}={values['accuracy'][-1]:.4f}"
            for label, values in metrics.items()
        ))

    sizes = {
        "float32": model.size_bytes(),
        "int8": compressed_size_bytes(compressed_by_bits["int8"]),
        "int6": int6_size_bytes(compressed_by_bits["int6"]),
        "int4": int4_size_bytes(compressed_by_bits["int4"]),
    }
    print(f"\n{name} precision comparison ({len(SEEDS)} seeds, "
          f"{(time.time() - started) / 60:.1f} min; weight storage only):")
    for label, values in metrics.items():
        acc = values["accuracy"]
        f1 = values["f1"]
        print(f"  {label:<7}: accuracy={sum(acc)/len(acc):.4f} +/-{ci_95(acc):.4f}  "
              f"macro-F1={sum(f1)/len(f1):.4f} +/-{ci_95(f1):.4f}  "
              f"size={sizes[label]} B")
        if label != "float32":
            for metric_name, baseline, candidate in (
                ("accuracy", metrics["float32"]["accuracy"], acc),
                ("macro-F1", metrics["float32"]["f1"], f1),
            ):
                result = tost_paired(baseline, candidate)
                verdict = "EQUIV" if result["equivalent"] else "NOT EQUIV"
                print(f"           TOST {metric_name}: {verdict}, "
                      f"diff={result['mean_diff']:+.4f}, "
                      f"CI=[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]")


if __name__ == "__main__":
    for dataset in DATASETS:
        run_precision_comparison(*dataset)
