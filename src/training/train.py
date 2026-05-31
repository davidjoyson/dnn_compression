import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def train(model, X, y, epochs=1, lr=1e-3, batch_size=64,
          X_val=None, y_val=None, num_classes=1):
    """
    Training loop for all experiments.
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
    val_history = {"loss": [], "acc": []} if X_val is not None else None

    for _ in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()

            epoch_loss += loss.detach().item()
            n_batches += 1

        history.append(epoch_loss / n_batches if n_batches > 0 else 0.0)

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_out = model(X_val)
                val_loss = loss_fn(val_out, y_val).item()
                if num_classes > 1:
                    val_acc = val_out.argmax(dim=1).eq(y_val.long()).float().mean().item()
                else:
                    val_acc = (val_out > 0.5).float().eq(y_val).float().mean().item()
            val_history["loss"].append(val_loss)
            val_history["acc"].append(val_acc)
            model.train()

    if X_val is not None:
        return history, val_history
    return history
