import torch
from torch.utils.data import default_collate
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from typing import List, Tuple

# Replace all advanced transforms with a simple pipeline
simple_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create your train_set and test_set with no fancy augmentation
train_set = CIFAR10(
    root="./data",
    train=True,
    transform=simple_transform,
    download=False  # Set to True only once to download, then False afterwards
)
test_set = CIFAR10(
    root="./data",
    train=False,
    transform=simple_transform,
    download=False
)

# If you don't want any batchwise augmentation, you can just do:
def collate_fn_no_augment(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    return default_collate(batch)
