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

    def size_bytes(self):
        return sum(p.nelement() * p.element_size() for p in self.parameters())


class LayerMatchedMLP(nn.Module):
    """
    Plain sequential MLP with the same per-stage widths as DendriticNetwork's
    trunk (fc1 -> branch/soma bottleneck -> fc2 -> out), but no branching —
    a single dense layer replaces the 8 parallel branches + soma.

    Isolates whether the branching/parallel-topology itself matters,
    independent of total parameter count: the branches+soma block is
    parameter-redundant by design (8 separate weight matrices converging to
    a narrow output), so this baseline has *fewer* total params than
    DendriticNetwork for the same shape — MLPBaseline (param-matched) is
    still the total-budget control; this is the per-layer-shape control.
    """
    def __init__(self, input_dim, hidden_neurons1, branches, hidden_neurons2, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        out_dim = max(1, num_classes)

        self.fc1 = nn.Linear(input_dim, hidden_neurons1)
        self.mid = nn.Linear(hidden_neurons1, branches)
        self.fc2 = nn.Linear(branches, hidden_neurons2)
        self.out = nn.Linear(hidden_neurons2, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.mid(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        if self.num_classes == 1:
            return torch.sigmoid(x)
        return x

    def size_bytes(self):
        return sum(p.nelement() * p.element_size() for p in self.parameters())

