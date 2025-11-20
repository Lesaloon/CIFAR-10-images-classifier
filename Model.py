import torch
import torch.nn as nn
from ConvBlock import ConvBlock
from SEBlock import SEBlock


class Model(nn.Module):
    """CNN for CIFAR‑10 with Conv blocks, GAP, and SE attention.

    Architecture summary (32×32 RGB input):
    - Feature extractor: 3× `ConvBlock` with MaxPool(2)
        Channels: 3 → 64 → 128 → 256
        Spatial sizes: 32 → 16 → 8 → 4
    - Global Average Pooling (GAP): reduces 4×4 to 1×1 per channel (256-dim vector)
    - Squeeze-and-Excitation (SEBlock): channel-wise attention with reduction=16
    - Classifier (MLP): 256 → 256 (ReLU + Dropout) → 10 logits

    Notes:
    - CrossEntropyLoss expects raw logits; no softmax in the model.
    - `self.output` is a legacy layer and is not used by `forward`.
    """

    def __init__(self):
        super(Model, self).__init__()
        # Feature extractor:
        # Each ConvBlock: Conv(3x3, pad=1) -> BatchNorm -> ReLU -> MaxPool(2)
        # Spatial progression for 32x32 input:
        # 32x32 (3ch) -> 16x16 (64ch) -> 8x8 (128ch) -> 4x4 (256ch)
        self.features = nn.Sequential(
            ConvBlock(3, 64),  # Block 1: 3→64 channels, spatial 32→16
            ConvBlock(64, 128),  # Block 2: 64→128 channels, spatial 16→8
            ConvBlock(128, 256),  # Block 3: 128→256 channels, spatial 8→4
        )
        self.se = SEBlock(256, reduction=16)  # Channel attention on 256 channels
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global AvgPool: (B,256,4,4)→(B,256,1,1)

        # Classifier head (MLP): produces 10 logits
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),  # 256-d vector → hidden 256
            nn.ReLU(),  # non-linearity
            nn.Dropout(0.5),  # regularization
            nn.Linear(256, 10),  # hidden → 10 class logits
        )
        self.output = nn.Linear(256, 10)  # legacy/unreferenced (not used in forward)

    def forward(self, x):
        x = self.features(x)  # (B,3,32,32) → (B,256,4,4)
        x = self.gap(x)  # (B,256,4,4) → (B,256,1,1)
        x = self.se(x)  # channel-wise reweighting (same shape)
        x = x.view(x.size(0), -1)  # flatten: (B,256,1,1) → (B,256)
        x = self.classifier(x)  # (B,256) → (B,10) logits
        return x
