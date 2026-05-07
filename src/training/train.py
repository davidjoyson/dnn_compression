import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
tqdm_config = {"colour": "white"}

def train(model, X, y, epochs=1, lr=1e-3, batch_size=64, use_tqdm=True):
    """
    Training loop for all experiments.
    - use_tqdm=True  → progress bar (default)
    - use_tqdm=False → no progress bar (for scaling experiments)
    """

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    model.train()
    last_loss = None

    for epoch in range(epochs):

        # Choose loop type
        if use_tqdm:
            loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, **tqdm_config)
        else:
            loop = loader

        for xb, yb in loop:
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()

            last_loss = loss.detach().item()

            # Only update progress bar when enabled
            if use_tqdm:
                loop.set_postfix(loss=last_loss)

    return last_loss
