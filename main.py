import matplotlib.pyplot as plt
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from CIFARDataset import CIFARDataset
from Model import Model
import numpy as np
import math
from torch.amp.grad_scaler import GradScaler  # unified AMP GradScaler
from torch.amp.autocast_mode import autocast  # unified AMP autocast


def train(
    model: Model,
    data: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    amp: bool = True,
    date: str = "",
) -> None:
    model.to(device)
    # Use new torch.amp GradScaler API (pass device string) to avoid deprecation warnings
    scaler = GradScaler("cuda", enabled=amp and device.type == "cuda")
    for epoch in range(epochs):
        model.train()
        currtime = datetime.now()
        current_loss = 0.0
        current_acc = 0.0
        size = len(data)

        for X, y, label in data:
            X = X.to(device).float()
            y = y.to(device).float()
            optimizer.zero_grad(set_to_none=True)

            # Use new torch.amp.autocast API (first arg is device string)
            with autocast("cuda", enabled=amp and device.type == "cuda"):
                output = model(X)
                loss = criterion(output, torch.argmax(y, dim=1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            current_loss += loss.item()
            current_acc += compute_accuracy(output, y)

        if scheduler is not None:
            scheduler.step()

        save_model(model, f"model/checkpoint/model_epoch_{epoch+1:03d}_{date}.pth")

        print(
            f"Epoch {epoch+1:03d}/{epochs:03d} - Loss: {current_loss/size:.6f} - Acc: {current_acc/size:.6f} - LR: {optimizer.param_groups[0]['lr']:.6f} - Time: {datetime.now() - currtime}"
        )


def save_model(model: Model, path: str) -> None:
    torch.save(model.state_dict(), path)


def compute_accuracy(pred: torch.Tensor, labels: torch.Tensor) -> float:
    pred_np = pred.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()
    score = 0.0
    for i in range(len(pred_np)):
        score += np.argmax(pred_np[i]) == np.argmax(labels_np[i])
    return score / pred_np.shape[0]


if __name__ == "__main__":
    train_dataset = CIFARDataset("CIFAR-10-images/train")
    test_dataset = CIFARDataset("CIFAR-10-images/test")
    print(f"Nombre d'exemples dans le jeu d'entrainement : {len(train_dataset)}")
    print(f"Nombre d'exemples dans le jeu de test : {len(test_dataset)}")
    # show first image of the training dataset
    # image, bin_label, label = train_dataset[0]
    # print(f"Label binaire : {bin_label}, Label entier : {label}")
    # plt.imshow(image.reshape(28, 28), cmap="gray")
    # plt.title(f"Label : {label}")
    # plt.show()
    traindata = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
    test = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )
    print(f"Nombre de batchs dans le jeu d'entrainement : {len(traindata)}")
    print(f"Nombre de batchs dans le jeu de test : {len(test)}")

    iterateur = iter(traindata)
    batch = next(iterateur)
    # X, y = batch[0][0].numpy(), batch[2][0].numpy()
    # imgplot = plt.imshow(X.reshape(28, 28), cmap="gray")
    # plt.title(f"Label : {y}")
    # plt.show()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Model()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    epochs = 500
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    train(
        model,
        traindata,
        optimizer,
        criterion,
        device,
        epochs,
        scheduler,
        amp=True,
        date=date,
    )
    torch.save(model.state_dict(), "model/final/model_" + date + ".pth")
