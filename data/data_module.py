import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from data.data_augmentation import (
    transforms_augmentation,
    transforms_no_augment,
    collate_fn_augment,
    collate_fn_no_augment,
)

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128, val_split=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage=None):
        # 1) Load the *full* training set with NO augmentation
        full_train_set = CIFAR10(
            root=self.data_dir,
            train=True,
            transform=transforms_no_augment,  # <- Start with no augmentation
            download=True,
        )

        # 2) Split into train/val
        train_size = int((1 - self.val_split) * len(full_train_set))
        val_size = len(full_train_set) - train_size
        self.train_set, self.val_set = random_split(full_train_set, [train_size, val_size])

        # 3) Override the transform *only* for the training subset
        self.train_set.dataset.transform = transforms_augmentation

        # 4) Test set remains unchanged
        self.test_set = CIFAR10(
            root=self.data_dir,
            train=False,
            transform=transforms_no_augment,
            download=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_augment,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_no_augment,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_no_augment,
            num_workers=4,
            persistent_workers=True,
        )
