import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms as transform

enum_labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class CIFARDataset(Dataset):
    def __init__(self, dataset_path):
        self.transform = transform.Compose(
            [
                transform.RandomHorizontalFlip(),
                transform.RandomCrop(32, padding=4),
                transform.ToTensor(),
            ]
        )
        self.dir = dataset_path
        # files are into folders named using their labels
        self.instancesFile = []
        for label in sorted(os.listdir(self.dir)):
            label_dir = os.path.join(self.dir, label)
            if os.path.isdir(label_dir):
                for filename in sorted(os.listdir(label_dir)):
                    self.instancesFile.append(os.path.join(label, filename))

        super().__init__()

    def __len__(self):
        return len(self.instancesFile)

    def __getitem__(self, idx):
        """
        Get the Item at index idx
        :param idx: index of the item
        """
        try:
            filename = os.path.join(self.dir, self.instancesFile[idx])
            # print(self.instancesFile[idx].split("\\"))
            X, y = [], enum_labels.index(
                self.instancesFile[idx].split("\\")[0]
            )  # get label from folder name

            # jpeg, get pixels values, 3 channels, resize to 32x32
            with Image.open(filename) as img:
                img = img.convert("RGB")
                img = img.resize((32, 32))
                img = self.transform(img)
                X = np.array(img)
            # one-hot encode the label ( 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 )
            binY = np.zeros(len(enum_labels))
            binY[y] = 1

            return np.array(X).reshape(3, 32, 32), binY, y
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            ## show the item that caused the error
            print(self.instancesFile[idx])
            raise
