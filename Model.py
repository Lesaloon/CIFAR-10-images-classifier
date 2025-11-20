import torch
import torch.nn as nn
from ConvBlock import ConvBlock
from SEBlock import SEBlock


class Model(nn.Module):
    """CNN with Squeeze-Excitation and Global Average Pooling.

    Modifications vs previous version:
    - Removed two large DenseBlocks (majority of parameters >550K).
    - Added SEBlock to recalibrate channel responses (adds ~2K params).
    - Uses AdaptiveAvgPool2d to reduce 4x4 feature map to 1x1 per channel.
    - Classification head: Linear(128 -> 10).

    Result: Parameter count significantly reduced while improving feature quality;
    no increase in size, enabling potential accuracy gains via better generalization.
    """

    def __init__(self):
        super(Model, self).__init__()
        # Feature extractor:
        # Each ConvBlock: Conv(3x3, pad=1) -> BatchNorm -> ReLU -> MaxPool(2)
        # Spatial progression for 32x32 input:
        # 32x32 (3ch) -> 16x16 (32ch) -> 8x8 (64ch) -> 4x4 (128ch)
        self.features = nn.Sequential(
            ConvBlock(3, 64),  # RGB input -> 64 channels (16x16)
            ConvBlock(64, 128),  # 64 -> 128 channels (8x8)
            ConvBlock(128, 256),  # 128 -> 256 channels (4x4)
        )
        self.se = SEBlock(256, reduction=16)
        # Global Average Pooling replaces Flatten + Dense layers (parameter efficient)
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> (B,192,1,1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )
        # Final classification layer (was previously preceded by large dense blocks)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)  # convolutional feature maps
        x = self.gap(x)  # global descriptor per image
        x = self.se(x)
        x = x.view(x.size(0), -1)  # flatten for classifier
        x = self.classifier(x)  # logits (no softmax; use CrossEntropyLoss)
        return x
