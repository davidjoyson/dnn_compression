import torch

def to_tensor(X, y=None):
    X = torch.tensor(X, dtype=torch.float32)
    if y is not None:
        y = torch.tensor(y, dtype=torch.float32)
    return X, y
