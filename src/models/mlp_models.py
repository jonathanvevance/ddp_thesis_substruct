"""Python file with MLP model classes."""

import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ScoringNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_layers = nn.ModuleList([
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        ])
        self.concat_mlp = nn.Linear(256, 1)

    def forward(self, x1, x2):

        for layer in self.shared_layers:
            x1 = layer(x1)

        for layer in self.shared_layers:
            x2 = layer(x2)

        return torch.sigmoid(self.concat_mlp(x1 + x2))

# TODO: research common (graph or just vector) aggregation strategies
# TODO: Ex. like there exists for attention mechanism