import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, h, w = x.size()
        # Squeeze: global average pooling
        y = x.view(b, c, -1).mean(dim=2)  # (B, C)
        # Excitation: bottleneck MLP
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))  # (B, C)
        y = y.view(b, c, 1, 1)  # (B, C, 1, 1)
        return x * y  # scale channels
