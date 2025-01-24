import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from data.data_augmentation import train_set, test_set, collate_fn_no_augment
import torchvision.transforms as transforms

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define basic transforms (no advanced aug here, or add them if you want)
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        # Download only once
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=True,
                transform=self.train_transform,
            )
            # Example: 45k for train, 5k for val
            self.train_data, self.val_data = random_split(full_train_dataset, [45000, 5000])

        if stage == "test" or stage is None:
            self.test_data = torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                transform=self.test_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )