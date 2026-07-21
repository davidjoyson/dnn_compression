import numpy as np
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


def per_class_stats_from_cm(cm, min_support=20):
    """One-vs-rest precision/recall/specificity/F1 per class, plus balanced
    accuracy (mean per-class recall), computed directly from an existing
    confusion matrix -- no re-evaluation needed.

    min_support: classes with fewer than this many true (test-set) examples
    are excluded from the "_supported" macro averages -- with only a
    handful of real examples, a class's F1/recall is mostly noise, and
    including it just distorts the macro average. The full, all-classes
    averages are still returned for reference."""
    cm = np.asarray(cm, dtype=float)
    total = cm.sum()
    precision, recall, specificity, f1, support = [], [], [], [], []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn
        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        specificity.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
        f1.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        support.append(int(cm[c, :].sum()))

    supported = [c for c in range(cm.shape[0]) if support[c] >= min_support]
    excluded = [c for c in range(cm.shape[0]) if support[c] < min_support]

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "support": support,
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "supported_classes": supported,
        "excluded_classes": excluded,
        "balanced_accuracy_supported": float(np.mean([recall[c] for c in supported])) if supported else None,
        "macro_f1_supported": float(np.mean([f1[c] for c in supported])) if supported else None,
    }


def predict_proba_multiclass(model, X, device=None):
    """Return softmax probabilities as a CPU numpy array (N, num_classes)."""
    import torch.nn.functional as F
    device = device or _DEVICE
    model = model.to(device)
    X = X.to(device)
    model.eval()
    with torch.no_grad():
        return F.softmax(model(X), dim=1).cpu().numpy()
