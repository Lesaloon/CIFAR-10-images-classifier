import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size), nn.ReLU(), nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.layers(x)
