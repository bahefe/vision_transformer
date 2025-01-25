import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.data_augmentation import (
    train_set,
    test_set,
    collate_fn_augment,
    collate_fn_no_augment,
)

from pytorch_lightning import LightningDataModule


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # If you're already creating `train_set` and `test_set` in data_augmentation.py`,
        # there's nothing else to do here. Just confirm there's no reference to
        # self.train_transform or self.test_transform.
        pass

    def train_dataloader(self):
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_augment,  # for MixUp/CutMix
            num_workers=0,
        )

    def val_dataloader(self):
        # If you don't have a separate val set, reuse test_set
        return DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_no_augment,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_no_augment,
            num_workers=0,
        )
