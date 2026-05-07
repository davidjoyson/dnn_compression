import torch.nn as nn
from .dendritic_layer import DendriticLayer

class DendriticNetwork(nn.Module):
    """
    Two dendritic layers + linear output.
    """
    def __init__(self, input_dim, hidden_neurons1=8, hidden_neurons2=4,
                 branches=4, hidden_per_branch=4):
        super().__init__()
        self.layer1 = DendriticLayer(input_dim, hidden_neurons1, branches, hidden_per_branch)
        self.layer2 = DendriticLayer(hidden_neurons1, hidden_neurons2, branches, hidden_per_branch)
        self.out = nn.Linear(hidden_neurons2, 1)

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        return self.out(h2).sigmoid()
