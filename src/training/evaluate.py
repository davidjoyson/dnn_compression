import torch

def evaluate(model, X, y, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    X = X.to(device)
    y = y.to(device)

    model.eval()
    with torch.no_grad():
        preds = (model(X) > 0.5).float()
        acc = (preds.eq(y).float().mean()).item()
    return acc
