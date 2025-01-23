import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torchvision
from data.data_augmentation import train_set, test_set, collate_fn_no_augment
import torchvision.transforms as T

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Use the entire train set without splitting
            self.train_data = torchvision.datasets.CIFAR10(
                self.data_dir, train=True, transform=self.transform
            )
        if stage == "test" or stage is None:
            self.test_data = torchvision.datasets.CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
            collate_fn=collate_fn_no_augment, 
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
            collate_fn=collate_fn_no_augment,
        )
