import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017). Down-weights easy/already-correct
    examples and focuses gradient on hard, boundary-case examples -- unlike
    class-weighting, which weights every example of a class equally regardless
    of how confidently it's already classified."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[targets]
        return loss.mean()


class TverskyLoss(nn.Module):
    """Multi-class Tversky loss. alpha weights false positives, beta weights
    false negatives (alpha=beta=0.5 reduces to Dice loss). Unlike class
    weighting/focal loss, this lets precision and recall be traded off
    explicitly rather than symmetrically -- set alpha > beta to penalize
    false positives more (raise precision), beta > alpha for the opposite."""
    def __init__(self, num_classes, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, self.num_classes).float()
        tp = (probs * targets_onehot).sum(dim=0)
        fp = (probs * (1 - targets_onehot)).sum(dim=0)
        fn = ((1 - probs) * targets_onehot).sum(dim=0)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()
