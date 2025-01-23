# data/data_module.py
import torch
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from .data_augmentation import train_set, test_set, collate_fn_augment, collate_fn_no_augment

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # Downloads CIFAR10 if needed
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # If stage is "fit" or None, just assign the entire train_set as train_data
        if stage in (None, "fit"):
            self.train_data = train_set  # NO validation split

        if stage in (None, "test"):
            self.test_data = test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn_no_augment,
        )

    # Remove or disable the val_dataloader
    # def val_dataloader(self):
    #     return None

    # Keep the test_dataloader if you want final testing
    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn_no_augment,
        )
