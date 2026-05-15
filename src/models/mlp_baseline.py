import torch.nn as nn


def param_matched_hidden(total_params, input_dim, num_classes=1):
    """Return hidden size H so that MLP param count ≈ total_params."""
    out_dim = max(1, num_classes)
    return max(1, round((total_params - out_dim) / (input_dim + 1 + out_dim)))


class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden=16, match_params=None, num_classes=1):
        if match_params is not None:
            hidden = param_matched_hidden(match_params, input_dim, num_classes)
        super().__init__()
        self.num_classes = num_classes
        out_dim = max(1, num_classes)
        layers = [nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)]
        if num_classes == 1:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def size_bytes(self):
        return sum(p.nelement() * p.element_size() for p in self.parameters())
