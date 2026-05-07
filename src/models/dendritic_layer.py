import torch
import torch.nn as nn
from .dendritic_neuron import DendriticNeuron

class DendriticLayer(nn.Module):
    def __init__(self, input_dim, num_neurons, num_branches, hidden_per_branch=4):
        super().__init__()
        self.neurons = nn.ModuleList([
            DendriticNeuron(input_dim, num_branches, hidden_per_branch)
            for _ in range(num_neurons)
        ])

    def forward(self, x):
        outs = [n(x) for n in self.neurons]
        return nn.functional.relu(torch.cat(outs, dim=1))
