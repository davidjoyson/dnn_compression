import copy
import numpy as np
import torch
import torch.nn.functional as F


def _capture_branch_acts(model, X, device="cpu"):
    """Run forward pass, return list of per-branch output activations (after ReLU)."""
    model = model.to(device).eval()
    X = X.to(device)
    acts = [None] * len(model.branches)

    def make_hook(i):
        def h(mod, inp, out):
            acts[i] = F.relu(out).detach().cpu()
        return h

    hooks = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(model.branches)]
    with torch.no_grad():
        model(X)
    for h in hooks:
        h.remove()
    return acts


def branch_weight_similarity(model):
    """(n_branches, n_branches) pairwise cosine similarity of flattened branch weights."""
    W = torch.stack([b.weight.data.cpu().flatten() for b in model.branches])
    W = F.normalize(W, dim=1)
    return (W @ W.T).numpy()


def branch_activation_correlation(model, X, device="cpu"):
    """(n_branches, n_branches) Pearson correlation of mean branch activations over X."""
    acts = _capture_branch_acts(model, X, device)
    B = np.array([a.mean(dim=1).numpy() for a in acts])  # (n_branches, N)
    return np.corrcoef(B)


def branch_quant_error(model_float, model_quant, X, device="cpu"):
    """Per-branch MSE between float32 and dequantized activations."""
    float_acts = _capture_branch_acts(model_float, X, device)
    quant_acts = _capture_branch_acts(model_quant, X, device)
    return [((f - q) ** 2).mean().item() for f, q in zip(float_acts, quant_acts)]


def compute_branch_diversity(model_float, model_quant, X, device="cpu"):
    """
    Compute all three branch diversity metrics.
    model_float: trained float32 DendriticNetwork
    model_quant: same model after decompress_model() (dequantized INT8 weights)
    X: calibration/training data tensor
    """
    return {
        "weight_similarity":      branch_weight_similarity(model_float),
        "activation_correlation": branch_activation_correlation(model_float, X, device),
        "quant_error_per_branch": branch_quant_error(model_float, model_quant, X, device),
    }
