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
        hidden_per_branch,
        use_soma=True,
        num_classes=1,
    ):
        super().__init__()
        self.use_soma = use_soma
        self.num_classes = num_classes
        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_neurons1)

        # Dendritic branches (shared topology)
        self.branches = nn.ModuleList([
            nn.Linear(hidden_neurons1, hidden_per_branch)
            for _ in range(branches)
        ])

        # Soma: collapses each branch's hidden_per_branch activations → one signal per branch
        self.soma = nn.Linear(branches * hidden_per_branch, branches) if use_soma else None

        # Layer 2
        fc2_in = branches if use_soma else branches * hidden_per_branch
        self.fc2 = nn.Linear(fc2_in, hidden_neurons2)

        # Output layer
        self.out = nn.Linear(hidden_neurons2, max(1, num_classes))

    def forward(self, x):

        # First layer
        x = F.relu(self.fc1(x))

        # Dendritic branches
        branch_outputs = []
        for b in self.branches:
            branch_outputs.append(F.relu(b(x)))

        # Concatenate branch outputs
        x = torch.cat(branch_outputs, dim=1)

        # Soma: one integrated signal per branch
        if self.use_soma:
            x = F.relu(self.soma(x))

        # Second layer
        x = F.relu(self.fc2(x))

        # Output
        x = self.out(x)
        if self.num_classes == 1:
            return torch.sigmoid(x)
        return x  # raw logits for CrossEntropyLoss

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
        fc1   = self.fc1
        b0    = self.branches[0]
        fc2   = self.fc2
        out   = self.out
        n     = len(self.branches)
        print(f"DendriticNetwork")
        print(f"  fc1     : Linear({fc1.in_features} → {fc1.out_features})")
        print(f"  branches: {n} × Linear({b0.in_features} → {b0.out_features})")
        if self.use_soma:
            s = self.soma
            print(f"  soma    : Linear({s.in_features} → {s.out_features})")
        print(f"  fc2     : Linear({fc2.in_features} → {fc2.out_features})")
        print(f"  out     : Linear({out.in_features} → {out.out_features})")
        print(f"  params  : {sum(p.numel() for p in self.parameters()):,}")
