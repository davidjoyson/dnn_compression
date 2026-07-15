"""Quick int4-only run across all 4 datasets."""
import time
import torch
from sklearn.model_selection import train_test_split

from src.loaders.load_har import load_har
from src.loaders.load_eeg import load_eeg
from src.loaders.load_hapt import load_hapt
from src.models.dendritic_network import DendriticNetwork
from src.training.train import train
from src.training.evaluate import evaluate, f1_eval
from src.compression.compression_pipeline import (
    compress_model_int4, decompress_model_int4, int4_size_bytes,
    compressed_size_bytes, compress_model, decompress_model,
)
from src.analysis.tost import ci_95, tost_paired

SEEDS = (42, 0, 7)
EPOCHS = 50
FINE_TUNE = 3

DATASETS = [
    ("EEG",  load_eeg,  3),
    ("HAPT", load_hapt, 12),
]


def run_int4(name, loader, num_classes):
    acc_u, acc_c, acc_i4 = [], [], []
    t0 = time.time()
    for seed in SEEDS:
        X_tr, y_tr, X_te, y_te = loader()
        X_tr, X_v, y_tr, y_v = train_test_split(X_tr, y_tr, test_size=0.1,
                                                  random_state=seed, stratify=y_tr)
        Xtr = torch.tensor(X_tr, dtype=torch.float32)
        ytr = torch.tensor(y_tr, dtype=torch.long)
        Xte = torch.tensor(X_te, dtype=torch.float32)
        yte = torch.tensor(y_te, dtype=torch.long)

        torch.manual_seed(seed)
        model = DendriticNetwork(input_dim=Xtr.shape[1], hidden_neurons1=64,
                                 hidden_neurons2=32, branches=8,
                                 hidden_per_branch=8, num_classes=num_classes)
        train(model, Xtr, ytr, epochs=EPOCHS, num_classes=num_classes)
        acc_u.append(evaluate(model, Xte, yte, num_classes=num_classes))

        orig = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Snowflake int8 (reference)
        c8 = compress_model(model, fine_tune_data=(Xtr, ytr), fine_tune_epochs=FINE_TUNE)
        decompress_model(c8, model)
        acc_c.append(evaluate(model, Xte, yte, num_classes=num_classes))

        # Int4
        model.load_state_dict(orig)
        c4 = compress_model_int4(model, fine_tune_data=(Xtr, ytr), fine_tune_epochs=FINE_TUNE)
        decompress_model_int4(c4, model)
        acc_i4.append(evaluate(model, Xte, yte, num_classes=num_classes))

        print(f"  {name} seed={seed}: u={acc_u[-1]:.4f}  int8={acc_c[-1]:.4f}  int4={acc_i4[-1]:.4f}")

    mean = lambda lst: sum(lst) / len(lst)
    ci   = lambda lst: ci_95(lst)
    tost = tost_paired(acc_u, acc_i4)

    sz_u  = model.size_bytes()
    sz_i4 = int4_size_bytes(c4)
    sz_i8 = compressed_size_bytes(c8)

    elapsed = time.time() - t0
    print(f"\n{name} ({len(SEEDS)} seeds, {elapsed/60:.1f} min):")
    print(f"  Uncompressed   : {mean(acc_u):.4f} ±{ci(acc_u):.4f}")
    print(f"  Snowflake int8 : {mean(acc_c):.4f} ±{ci(acc_c):.4f}  [{sz_u}B -> {sz_i8}B = {sz_u/sz_i8:.1f}x]")
    print(f"  Snowflake int4 : {mean(acc_i4):.4f} ±{ci(acc_i4):.4f}  [{sz_u}B -> {sz_i4}B = {sz_u/sz_i4:.1f}x]")
    equiv = "EQUIV" if tost["equivalent"] else "NOT EQUIV"
    print(f"  TOST int4      : {equiv}  diff={tost['mean_diff']:+.4f}  CI=[{tost['ci_low']:+.4f}, {tost['ci_high']:+.4f}]")


for ds in DATASETS:
    run_int4(*ds)
