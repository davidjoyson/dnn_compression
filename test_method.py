"""Test a single compression method against a saved uncompressed checkpoint.

Skips retraining the base model and skips the other 7 compression methods —
use this while iterating on one method instead of a full `main.py` run.

Usage:
  python test_method.py --dataset ecg --method snowflake_static
"""
import argparse
import torch

from src.loaders.load_har import load_har
from src.loaders.load_ecg_patient_split import load_ecg_patient_split
from src.loaders.load_eeg import load_eeg
from src.loaders.load_hapt import load_hapt
from src.models.dendritic_network import DendriticNetwork
from src.training.evaluate import evaluate, f1_eval
from src.compression.compression_pipeline import (
    compress_model, decompress_model, compressed_size_bytes,
    compress_model_global,
    compress_model_dynamic, dynamic_model_size_bytes,
    compress_model_static, static_model_size_bytes,
    compress_model_snowflake_static,
    compress_model_per_channel, decompress_model_per_channel, per_channel_size_bytes,
    compress_model_qat,
    compress_model_mixed, mixed_model_size_bytes,
    compress_model_int4, decompress_model_int4, int4_size_bytes,
)

LOADERS = {"har": load_har, "ecg": load_ecg_patient_split, "eeg": load_eeg, "hapt": load_hapt}
NUM_CLASSES = {"har": 6, "ecg": 5, "eeg": 3, "hapt": 12}
METHODS = ["snowflake", "global", "dynamic", "static", "snowflake_static",
           "perchan", "qat", "mixed", "int4"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(LOADERS))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--model-dir", default=None, help="Defaults to models/<dataset>")
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    args = parser.parse_args()

    model_dir = args.model_dir or f"models/{args.dataset}"
    num_classes = NUM_CLASSES[args.dataset]
    ft = args.fine_tune_epochs

    X_tr_np, y_tr_np, X_te_np, y_te_np = LOADERS[args.dataset]()
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_np, dtype=torch.long)
    X_te = torch.tensor(X_te_np, dtype=torch.float32)
    y_te = torch.tensor(y_te_np, dtype=torch.long)

    model = DendriticNetwork(input_dim=X_tr.shape[1], hidden_neurons1=64,
                             hidden_neurons2=32, branches=8, hidden_per_branch=8,
                             num_classes=num_classes)
    model.load_state_dict(torch.load(f"{model_dir}/dendritic_uncompressed.pt"))

    size_u = model.size_bytes()
    acc_u = evaluate(model, X_te, y_te, num_classes=num_classes)
    print(f"Uncompressed     : acc={acc_u:.4f}  size={size_u}B")

    m = args.method
    if m == "snowflake":
        c = compress_model(model, fine_tune_data=(X_tr, y_tr), fine_tune_epochs=ft)
        decompress_model(c, model)
        size_c = compressed_size_bytes(c)
        acc = evaluate(model, X_te, y_te, num_classes=num_classes)
        f1 = f1_eval(model, X_te, y_te, num_classes=num_classes)
    elif m == "global":
        c = compress_model_global(model, fine_tune_data=(X_tr, y_tr), fine_tune_epochs=ft)
        decompress_model(c, model)
        size_c = compressed_size_bytes(c)
        acc = evaluate(model, X_te, y_te, num_classes=num_classes)
        f1 = f1_eval(model, X_te, y_te, num_classes=num_classes)
    elif m == "dynamic":
        mq = compress_model_dynamic(model)
        size_c = dynamic_model_size_bytes(mq)
        acc = evaluate(mq, X_te, y_te, num_classes=num_classes, device="cpu")
        f1 = f1_eval(mq, X_te, y_te, num_classes=num_classes, device="cpu")
    elif m == "static":
        mq = compress_model_static(model, calibration_data=(X_tr, y_tr))
        size_c = static_model_size_bytes(mq)
        acc = evaluate(mq, X_te, y_te, num_classes=num_classes, device="cpu")
        f1 = f1_eval(mq, X_te, y_te, num_classes=num_classes, device="cpu")
    elif m == "snowflake_static":
        mq = compress_model_snowflake_static(model, calibration_data=(X_tr, y_tr))
        size_c = static_model_size_bytes(mq)
        acc = evaluate(mq, X_te, y_te, num_classes=num_classes, device="cpu")
        f1 = f1_eval(mq, X_te, y_te, num_classes=num_classes, device="cpu")
    elif m == "perchan":
        c = compress_model_per_channel(model)
        decompress_model_per_channel(c, model)
        size_c = per_channel_size_bytes(c)
        acc = evaluate(model, X_te, y_te, num_classes=num_classes)
        f1 = f1_eval(model, X_te, y_te, num_classes=num_classes)
    elif m == "qat":
        mq = compress_model_qat(model, train_data=(X_tr, y_tr), epochs=ft, num_classes=num_classes)
        size_c = static_model_size_bytes(mq)
        acc = evaluate(mq, X_te, y_te, num_classes=num_classes, device="cpu")
        f1 = f1_eval(mq, X_te, y_te, num_classes=num_classes, device="cpu")
    elif m == "mixed":
        mq = compress_model_mixed(model, calibration_data=(X_tr, y_tr))
        size_c = mixed_model_size_bytes(mq)
        acc = evaluate(mq, X_te, y_te, num_classes=num_classes, device="cpu")
        f1 = f1_eval(mq, X_te, y_te, num_classes=num_classes, device="cpu")
    elif m == "int4":
        c = compress_model_int4(model, fine_tune_data=(X_tr, y_tr), fine_tune_epochs=ft)
        decompress_model_int4(c, model)
        size_c = int4_size_bytes(c)
        acc = evaluate(model, X_te, y_te, num_classes=num_classes)
        f1 = f1_eval(model, X_te, y_te, num_classes=num_classes)

    ratio = size_u / size_c if size_c else float("nan")
    print(f"{m:16s} : acc={acc:.4f}  f1={f1:.4f}  delta={acc - acc_u:+.4f}  "
          f"size={size_u}B -> {size_c}B  ratio={ratio:.2f}x")


if __name__ == "__main__":
    main()
