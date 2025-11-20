# TP2 — Improved CNN for CIFAR‑10 (ESIEA)

Convolutional neural network implemented in PyTorch to classify CIFAR‑10 images (RGB 32×32, 10 classes). This work was carried out at ESIEA. The architecture was refined to improve accuracy and generalization while REDUCING total parameters.

## Overview

- Dataset: CIFAR‑10 (10 object classes, balanced)
- Input shape: `3 × 32 × 32`
- Output: Raw logits (`10` classes) for `CrossEntropyLoss`
- Key goals: Accuracy + efficiency (compact parameter count ~97K)

## Current Architecture (Model.py)

Feature extractor (three `ConvBlock`s):

- Each `ConvBlock`: `Conv2d(3×3, stride=1, pad=1)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2)`
- Channel progression: `3 → 32 → 64 → 128`
- Spatial progression: `32×32 → 16×16 → 8×8 → 4×4`

Attention & pooling:

- `SEBlock(128, reduction=16)`: Channel-wise squeeze & excitation (global avg pool → bottleneck FC → gating) adds ~2K params, improves channel discrimination.
- `AdaptiveAvgPool2d(1)`: Replaces large dense layers by producing a single global descriptor per channel (`128` features).

Classification head:

- `Linear(128 → 10)` for final logits.

Removed components from earlier version:

- Previous dense head (`2048 → 256 → 128`) with Dropout has been eliminated (it dominated parameter count and risked overfitting).

### Parameter Summary (approx.)

| Component            |   Params |
| -------------------- | -------: |
| 3 Conv2d + BN blocks |   ~93.7K |
| SEBlock              |    ~2.2K |
| Linear head          |    ~1.3K |
| **Total**            | **~97K** |

Previous version was >650K parameters (dense layers ~557K); new design is >6× smaller.

## Training Configuration (main.py)

- Optimizer: `AdamW(lr=1e-3, weight_decay=0.01)`
- Loss: `CrossEntropyLoss(label_smoothing=0.1)` (label smoothing mitigates overconfidence)
- Scheduler: `CosineAnnealingLR(T_max=epochs)` for smooth LR decay
- Mixed Precision: AMP (`GradScaler` + `autocast`) when CUDA available (faster training & lower memory)
- Batch size: `64`
- Epochs: `50` (adjustable)
- Device auto-select: GPU if available else CPU

## Data Pipeline

`CIFARDataset.py` loads files from:

- `CIFAR-10-images/train`
- `CIFAR-10-images/test`

Each sample yields:

- Image tensor: `3 × 32 × 32`
- One‑hot encoded label (training loop converts with `argmax` for loss calculation)

## Rationale for Changes

- Global Average Pooling + attention reduces parameters and overfitting risk.
- SEBlock introduces lightweight channel recalibration for higher representational power.
- Label smoothing + AdamW + cosine schedule improve generalization and training stability.
- Mixed precision speeds up training without accuracy loss.

## How to Run

### Setup

```bash
# 1) Create and activate a virtual environment (Windows Git Bash)
python -m venv .venv
source .venv/Scripts/activate

# 2) Install PyTorch following the official guide (choose your OS/CUDA)
#    https://pytorch.org/get-started/locally/
#    (Make sure to include torchvision in the selected command.)

# 3) Install other dependencies
python -m pip install numpy matplotlib pillow

# 4) Get the CIFAR-10-images dataset structure inside CIFAR-10-images-classifier
cd CIFAR-10-images-classifier
git clone git@github.com:YoongiKim/CIFAR-10-images.git

# 5) Run the program
python main.py
```

## Possible Next Improvements

- Add data augmentation (RandomCrop with padding, HorizontalFlip, ColorJitter)
- Introduce validation split + early stopping
- Experiment with smaller reduction factor in SEBlock (e.g., 8) or add stochastic depth

## Files

- `Model.py`: Current compact CNN with SEBlock + GAP
- `ConvBlock.py`: Convolutional building block
- `DenseBlock.py`: (Legacy) no longer used in model
- `CIFARDataset.py`: Dataset loader
- `main.py`: Training loop & configuration

## Attribution

This project is part of coursework completed at ESIEA.
