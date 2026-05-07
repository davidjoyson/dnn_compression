import torch
import torch.nn as nn
import torch.nn.functional as F

class DendriticNeuron(nn.Module):
    """
    Advanced dendritic neuron:
    - multiple branches
    - per-branch nonlinear transform
    - learned branch gating
    - soma integration
    """
    def __init__(self, input_dim, num_branches, hidden_per_branch=4):
        super().__init__()
        self.num_branches = num_branches

        # Each branch: input_dim -> hidden_per_branch -> scalar
        self.branch_linear1 = nn.Parameter(torch.randn(num_branches, input_dim, hidden_per_branch))
        self.branch_bias1   = nn.Parameter(torch.zeros(num_branches, hidden_per_branch))
        self.branch_linear2 = nn.Parameter(torch.randn(num_branches, hidden_per_branch, 1))
        self.branch_bias2   = nn.Parameter(torch.zeros(num_branches, 1))

        # Gating for each branch (how much each branch contributes)
        self.branch_gate = nn.Parameter(torch.ones(num_branches))

        # Soma bias
        self.soma_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [batch, input_dim]
        # branch 1st layer: [branches, batch, hidden]
        x_exp = x.unsqueeze(0)  # [1, batch, input_dim]
        h1 = torch.tanh(torch.einsum("bni,bij->bnj", x_exp, self.branch_linear1) + self.branch_bias1.unsqueeze(1))
        # branch 2nd layer: [branches, batch, 1]
        h2 = torch.tanh(torch.einsum("bnj,bjk->bnk", h1, self.branch_linear2) + self.branch_bias2.unsqueeze(1))
        # squeeze last dim -> [branches, batch]
        h2 = h2.squeeze(-1)

        # apply gating: [branches, batch] * [branches] -> [branches, batch]
        gated = h2 * self.branch_gate.unsqueeze(1)

        # soma integration: sum over branches -> [batch]
        soma = gated.sum(dim=0) + self.soma_bias
        return torch.tanh(soma).unsqueeze(1)
