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


def branch_weight_spread(model):
    """Per-branch weight std and range (max-min) -- tests the 'narrow,
    independently-calibrated distribution' claim directly (one number per
    branch), rather than just how similar branches are to each other."""
    stds, ranges = [], []
    for b in model.branches:
        w = b.weight.data.flatten()
        stds.append(w.std().item())
        ranges.append((w.max() - w.min()).item())
    return {"std": stds, "range": ranges}


def layer_matched_control_spread(layer_matched_model):
    """Per-output-row weight std/range of LayerMatchedMLP's `mid` layer --
    the structurally equivalent single-layer control for
    branch_weight_spread(): same input width, same row count (one row per
    branch-equivalent unit), same position in the network (right after
    fc1+ReLU) -- so it's an apples-to-apples comparison, not just a
    different-shaped baseline."""
    w = layer_matched_model.mid.weight.data
    stds = [row.std().item() for row in w]
    ranges = [(row.max() - row.min()).item() for row in w]
    return {"std": stds, "range": ranges}


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


def branch_saturation_rate(model, X_train, X_test, device="cpu"):
    """
    Per-branch fraction of test-time activations falling outside the
    calibration (training) range. Static/Snowflake+Static/QAT calibrate a
    fixed activation scale from training data and reuse it at test time —
    this measures what fraction of real test activations that fixed range
    would actually clip/saturate. Unlike weight saturation under Snowflake
    (trivially ~0, since the per-layer scale is defined by the max weight
    itself), this reflects genuine potential information loss at inference.
    """
    train_acts = _capture_branch_acts(model, X_train, device)
    test_acts = _capture_branch_acts(model, X_test, device)
    rates = []
    for tr, te in zip(train_acts, test_acts):
        lo, hi = tr.min().item(), tr.max().item()
        rates.append(((te < lo) | (te > hi)).float().mean().item())
    return rates


def compute_branch_diversity(model_float, model_quant, X, device="cpu", X_test=None):
    """
    Compute branch diversity metrics.
    model_float: trained float32 DendriticNetwork
    model_quant: same model after decompress_model() (dequantized INT8 weights)
    X: calibration/training data tensor
    X_test: optional held-out test data — if given, also computes per-branch
            saturation rate (fraction of test activations outside the range
            seen in X)
    """
    result = {
        "weight_similarity":      branch_weight_similarity(model_float),
        "activation_correlation": branch_activation_correlation(model_float, X, device),
        "quant_error_per_branch": branch_quant_error(model_float, model_quant, X, device),
        "weight_spread":          branch_weight_spread(model_float),
    }
    if X_test is not None:
        result["saturation_rate_per_branch"] = branch_saturation_rate(model_float, X, X_test, device)
    return result
