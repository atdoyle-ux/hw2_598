import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        in_size = 2
        h_size = 128
        out_size = 4
        self.network = nn.Sequential(
            nn.Linear(in_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: [B, 2] (row, col) normalized to [0..1]
        return self.network(x)