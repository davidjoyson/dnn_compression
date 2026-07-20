import torch
import torch.nn.functional as F


def output_divergence(model_float, model_quant, X, num_classes, device="cpu"):
    """
    Compare a compressed model's output directly against the float32
    reference's output on the same inputs — not against ground-truth labels.
    Two models can have identical downstream accuracy while producing very
    different logits/confidence; this measures that gap directly (the
    "true inference precision" question, distinct from accuracy/F1).
    """
    mf = model_float.to(device).eval()
    mq = model_quant.to(device).eval()
    Xd = X.to(device)

    with torch.no_grad():
        out_f = mf(Xd)
        out_q = mq(Xd)

    logit_mse = ((out_f - out_q) ** 2).mean().item()
    cosine_sim = F.cosine_similarity(out_f, out_q, dim=1).mean().item()

    if num_classes > 1:
        p_f = F.softmax(out_f, dim=1)
        p_q = F.softmax(out_q, dim=1)
        kl_div = F.kl_div((p_q + 1e-12).log(), p_f, reduction="batchmean").item()
        pred_flip_rate = (out_f.argmax(dim=1) != out_q.argmax(dim=1)).float().mean().item()
    else:
        kl_div = None
        pred_flip_rate = ((out_f > 0.5) != (out_q > 0.5)).float().mean().item()

    return {
        "logit_mse":         logit_mse,
        "cosine_similarity": cosine_sim,
        "kl_divergence":     kl_div,
        "pred_flip_rate":    pred_flip_rate,
    }
