"""Python file with MLP model classes."""

import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
