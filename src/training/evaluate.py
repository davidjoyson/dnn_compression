import torch

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, X, y, num_classes=1, device=None):
    device = device or _DEVICE
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    model.eval()
    with torch.no_grad():
        if num_classes > 1:
            preds = model(X).argmax(dim=1)
            acc = preds.eq(y.long()).float().mean().item()
        else:
            preds = (model(X) > 0.5).float()
            acc = preds.eq(y).float().mean().item()
    return acc


def mse_score(model, X, y, device=None):
    """MSE between sigmoid probabilities and true labels."""
    device = device or _DEVICE
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    model.eval()
    with torch.no_grad():
        return ((model(X) - y) ** 2).mean().item()


def f1_eval(model, X, y, num_classes=1, device=None):
    from sklearn.metrics import f1_score
    device = device or _DEVICE
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    model.eval()
    with torch.no_grad():
        if num_classes > 1:
            preds = model(X).argmax(dim=1).cpu().numpy()
            return float(f1_score(y.cpu().numpy(), preds, average="macro", zero_division=0))
        else:
            preds = (model(X) > 0.5).float().cpu().numpy().ravel()
            return float(f1_score(y.cpu().numpy().ravel(), preds, zero_division=0))


def confusion_matrix_eval(model, X, y, num_classes=1, device=None):
    from sklearn.metrics import confusion_matrix
    device = device or _DEVICE
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    model.eval()
    with torch.no_grad():
        if num_classes > 1:
            preds = model(X).argmax(dim=1).cpu().numpy()
            labels = y.cpu().numpy()
        else:
            preds = (model(X) > 0.5).float().cpu().numpy().ravel()
            labels = y.cpu().numpy().ravel()
    return confusion_matrix(labels, preds)


def predict_proba_multiclass(model, X, device=None):
    """Return softmax probabilities as a CPU numpy array (N, num_classes)."""
    import torch.nn.functional as F
    device = device or _DEVICE
    model = model.to(device)
    X = X.to(device)
    model.eval()
    with torch.no_grad():
        return F.softmax(model(X), dim=1).cpu().numpy()
