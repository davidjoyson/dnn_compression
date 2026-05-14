import torch.nn as nn


def param_matched_hidden(total_params, input_dim):
    """Return the MLP hidden size H so that (input_dim*H + H + H + 1) ≈ total_params."""
    return max(1, round((total_params - 1) / (input_dim + 2)))


class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden=16, match_params=None):
        if match_params is not None:
            hidden = param_matched_hidden(match_params, input_dim)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    def size_bytes(self):
        return sum(p.nelement() * p.element_size() for p in self.parameters())
