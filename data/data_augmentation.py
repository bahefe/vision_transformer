import torch
from torch.utils.data import default_collate
from torchvision.transforms import v2
from typing import List, Tuple

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

transforms_augmentation = v2.Compose([
    v2.Resize((32, 32)),
    v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10),
    v2.RandAugment(num_ops=2),
    v2.RandomErasing(0.15),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

transforms_no_augment = v2.Compose([
    v2.Resize((32, 32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])

# Batchwise augmentation (CutMix / MixUp)
cutmix_or_mixup = v2.RandomChoice([v2.CutMix(num_classes=10), v2.MixUp(num_classes=10)])

def collate_fn_augment(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply CutMix or MixUp to a batch."""
    return cutmix_or_mixup(*default_collate(batch))

def collate_fn_no_augment(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """No batchwise augmentation."""
    return default_collate(batch)
