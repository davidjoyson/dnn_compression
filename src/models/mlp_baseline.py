import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Hidden layer
        self.fc1 = nn.Linear(input_dim, hidden)

        # Output layer
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        # Hidden layer
        x = F.relu(self.fc1(x))

        # Output
        x = self.out(x)
        if self.num_classes == 1:
            return torch.sigmoid(x)
        return x

    # ---------------------------------------------------------
    # Size in bytes (for uncompressed model)
    # ---------------------------------------------------------
    def size_bytes(self):
        total = 0
        for p in self.parameters():
            total += p.nelement() * p.element_size()
        return total

    # ---------------------------------------------------------
    # Print architecture summary
    # ---------------------------------------------------------
    def print_arch(self):
        fc1 = self.fc1
        out = self.out
        print(f"MLPBaseline")
        print(f"  fc1 : Linear({fc1.in_features} → {fc1.out_features})")
        print(f"  out : Linear({out.in_features} → {out.out_features})")
        print(f"  params: {sum(p.numel() for p in self.parameters()):,}")
