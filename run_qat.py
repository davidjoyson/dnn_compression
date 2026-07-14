"""
Save QAT models for all 4 datasets using the already-trained uncompressed weights.
Outputs: models/<dataset>/dendritic_qat.pt
"""
import os
import time
import torch

from src.models.dendritic_network import DendriticNetwork
from src.loaders.load_har  import load_har
from src.loaders.load_ecg  import load_ecg
from src.loaders.load_eeg  import load_eeg
from src.loaders.load_hapt import load_hapt
from src.compression.compression_pipeline import compress_model_qat, static_model_size_bytes
from src.training.evaluate import evaluate

DATASETS = {
    "har":  (load_har,  561,  6),
    "ecg":  (load_ecg,  187,  5),
    "eeg":  (load_eeg,  2548, 3),
    "hapt": (load_hapt, 561,  12),
}

QAT_EPOCHS = 3
SEED = 42

torch.manual_seed(SEED)

for name, (loader, input_dim, num_classes) in DATASETS.items():
    model_dir = os.path.join("models", name)
    out_path  = os.path.join(model_dir, "dendritic_qat.pt")
    base_path = os.path.join(model_dir, "dendritic_uncompressed.pt")

    if not os.path.exists(base_path):
        print(f"[skip] {name}: {base_path} not found — run main.py first")
        continue

    print(f"\n[{name}] loading trained weights + data ...")
    X_tr_np, y_tr_np, X_te_np, y_te_np = loader()
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_np, dtype=torch.long)
    X_te = torch.tensor(X_te_np, dtype=torch.float32)
    y_te = torch.tensor(y_te_np, dtype=torch.long)

    model = DendriticNetwork(input_dim=input_dim, hidden_neurons1=64,
                             hidden_neurons2=32, branches=8,
                             hidden_per_branch=8, num_classes=num_classes)
    model.load_state_dict(torch.load(base_path, map_location="cpu"))

    t0 = time.time()
    print(f"[{name}] running QAT ({QAT_EPOCHS} epochs) ...")
    m_qat = compress_model_qat(model, (X_tr, y_tr),
                                epochs=QAT_EPOCHS, num_classes=num_classes)
    elapsed = time.time() - t0

    acc = evaluate(m_qat, X_te, y_te, num_classes=num_classes, device="cpu")
    size_kb = static_model_size_bytes(m_qat) / 1024

    torch.save(m_qat, out_path)
    print(f"[{name}] done in {elapsed:.0f}s  acc={acc:.4f}  size={size_kb:.1f}KB  -> {out_path}")
