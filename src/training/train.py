import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
tqdm_config = {"colour": "white"}


def train(model, X, y, epochs=1, lr=1e-3, batch_size=64, use_tqdm=True,
          X_val=None, y_val=None, num_classes=1):
    """
    Training loop for all experiments.
    - use_tqdm=True  → progress bar (default)
    - use_tqdm=False → no progress bar (for scaling experiments)
    - X_val / y_val  → if provided, computes val loss per epoch and returns
                        (train_history, val_history) instead of train_history alone
    - num_classes>1  → uses CrossEntropyLoss; y must be class indices (long)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if num_classes > 1:
        loss_fn = nn.CrossEntropyLoss()
        y = y.long()
        if y_val is not None:
            y_val = y_val.long()
    else:
        loss_fn = nn.BCELoss()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        pin_memory=(device.type == "cuda"), num_workers=0)

    if X_val is not None:
        X_val = X_val.to(device)
        y_val = y_val.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    history = []
    val_history = [] if X_val is not None else None

    for epoch in range(epochs):
        if use_tqdm:
            loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, **tqdm_config)
        else:
            loop = loader

        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()

            epoch_loss += loss.detach().item()
            n_batches += 1

            if use_tqdm:
                loop.set_postfix(loss=epoch_loss / n_batches)

        history.append(epoch_loss / n_batches if n_batches > 0 else 0.0)

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val), y_val).item()
            val_history.append(val_loss)
            model.train()

    if X_val is not None:
        return history, val_history
    return history
