"""Python file with MLP model classes."""

import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ScoringNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.ReLU(),
        ])

    def forward(self, x1, x2):

        for layer in self.layers:
            x1 = layer(x1)

        for layer in self.layers:
            x2 = layer(x2)

        return torch.softmax(0.5 * (x1 + x2), dim = 1)

# TODO: research common (graph or just vector) aggregation strategies
# TODO: Ex. like there exists for attention mechanism