"""
Standalone test: does Focal Loss or Tversky Loss beat plain CrossEntropy on
ECG patient-split F1/balanced accuracy? Not wired into the main pipeline --
just a quick comparison to decide whether either is worth adopting properly.

Usage:
  python test_loss_functions.py
"""
import time
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.loaders.load_ecg_patient_split import load_ecg_patient_split
from src.models.dendritic_network import DendriticNetwork
from src.training.evaluate import evaluate, f1_eval, confusion_matrix_eval, per_class_stats_from_cm
from src.training.losses import FocalLoss, TverskyLoss

SEED = 42
EPOCHS = 50
NUM_CLASSES = 5
CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]


def train_with_loss(loss_fn, X_train, y_train, X_val, y_val, label):
    torch.manual_seed(SEED)
    model = DendriticNetwork(
        input_dim=X_train.shape[1], hidden_neurons1=64, hidden_neurons2=32,
        branches=8, hidden_per_branch=8, num_classes=NUM_CLASSES,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    Xd, yd = X_train.to(device), y_train.to(device)
    Xv, yv = X_val.to(device), y_val.to(device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xd, yd), batch_size=256, shuffle=True
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_acc = model(Xv).argmax(dim=1).eq(yv).float().mean().item()
            print(f"    [{label}] epoch {epoch + 1:>3}/{EPOCHS}  loss={epoch_loss / len(loader):.4f}  val_acc={val_acc:.4f}")
    return model


def report(model, X_test, y_test, label):
    acc = evaluate(model, X_test, y_test, num_classes=NUM_CLASSES)
    f1 = f1_eval(model, X_test, y_test, num_classes=NUM_CLASSES)
    cm = confusion_matrix_eval(model, X_test, y_test, num_classes=NUM_CLASSES)
    stats = per_class_stats_from_cm(cm)
    print(f"\n=== {label} ===")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro F1 (all): {f1:.4f}")
    print(f"  Balanced Acc  : {stats['balanced_accuracy']:.4f}")
    if stats["excluded_classes"]:
        print(f"  Macro F1 (supported, excl. n<20)     : {stats['macro_f1_supported']:.4f}")
        print(f"  Balanced Acc (supported, excl. n<20) : {stats['balanced_accuracy_supported']:.4f}")
    print(f"  Per-class precision / recall / support:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:<20s}: {stats['precision'][i]:.3f} / {stats['recall'][i]:.3f} / n={stats['support'][i]}")
    return {"acc": acc, "f1": f1, "balanced_acc": stats["balanced_accuracy"],
            "f1_supported": stats["macro_f1_supported"]}


def main():
    print("Loading ECG patient-split data (balance=False, current default)...")
    X_train_np, y_train_np, X_test_np, y_test_np = load_ecg_patient_split(balance=False)

    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        X_train_np, y_train_np, test_size=0.1, random_state=SEED, stratify=y_train_np
    )
    X_train = torch.tensor(X_tr_np, dtype=torch.float32)
    y_train = torch.tensor(y_tr_np, dtype=torch.long)
    X_val   = torch.tensor(X_val_np, dtype=torch.float32)
    y_val   = torch.tensor(y_val_np, dtype=torch.long)
    X_test  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test  = torch.tensor(y_test_np, dtype=torch.long)

    results = {}

    t0 = time.time()
    model_ce = train_with_loss(nn.CrossEntropyLoss(), X_train, y_train, X_val, y_val, "CE baseline")
    results["CE (baseline)"] = report(model_ce, X_test, y_test, "CrossEntropy (baseline)")
    print(f"  Time: {time.time() - t0:.1f}s")

    t0 = time.time()
    model_focal = train_with_loss(FocalLoss(gamma=2.0), X_train, y_train, X_val, y_val, "Focal")
    results["Focal"] = report(model_focal, X_test, y_test, "Focal Loss (gamma=2.0)")
    print(f"  Time: {time.time() - t0:.1f}s")

    t0 = time.time()
    model_tversky = train_with_loss(
        TverskyLoss(num_classes=NUM_CLASSES, alpha=0.7, beta=0.3),
        X_train, y_train, X_val, y_val, "Tversky",
    )
    results["Tversky"] = report(model_tversky, X_test, y_test, "Tversky Loss (alpha=0.7, beta=0.3)")
    print(f"  Time: {time.time() - t0:.1f}s")

    print("\n=== Summary ===")
    print(f"{'Loss':<15s} {'Acc':>8s} {'F1 (all)':>10s} {'BalAcc':>8s} {'F1 (supp)':>10s}")
    for name, r in results.items():
        print(f"{name:<15s} {r['acc']:>8.4f} {r['f1']:>10.4f} {r['balanced_acc']:>8.4f} {r['f1_supported']:>10.4f}")


if __name__ == "__main__":
    main()
