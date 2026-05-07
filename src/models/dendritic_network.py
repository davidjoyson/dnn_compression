import torch
import torch.nn as nn
import torch.nn.functional as F


class DendriticNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_neurons1,
        hidden_neurons2,
        branches,
        hidden_per_branch
    ):
        super().__init__()
        self.branch_weights = None
        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_neurons1)

        # Dendritic branches (shared topology)
        self.branches = nn.ModuleList([
            nn.Linear(hidden_neurons1, hidden_per_branch)
            for _ in range(branches)
        ])

        # Layer 2
        self.fc2 = nn.Linear(branches * hidden_per_branch, hidden_neurons2)

        # Output layer
        self.out = nn.Linear(hidden_neurons2, 1)

    # ---------------------------------------------------------
    # REQUIRED FOR PYTORCH
    # ---------------------------------------------------------
    def forward(self, x):

        # First layer
        x = F.relu(self.fc1(x))

        # Dendritic branches
        branch_outputs = []
        for b in self.branches:
            branch_outputs.append(F.relu(b(x)))

        # Concatenate branch outputs
        x = torch.cat(branch_outputs, dim=1)

        # Second layer
        x = F.relu(self.fc2(x))

        # Output
        return torch.sigmoid(self.out(x))

    # ---------------------------------------------------------
    # Size in bytes (for uncompressed model)
    # ---------------------------------------------------------
    def size_bytes(self):
        total = 0
        for p in self.parameters():
            total += p.nelement() * p.element_size()
        return total
