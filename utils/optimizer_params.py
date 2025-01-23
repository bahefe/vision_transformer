# utils/optimizer_params.py

import math
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

def create_adam_linear_warmup(
    model_params,
    base_lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    warmup_epochs=5,
    total_epochs=100
):
    """
    Returns:
        optimizer, scheduler_dict
    - Optimizer: Adam with the specified betas and weight decay
    - Scheduler (dict form for PyTorch Lightning):
        * Linear LR warmup from 0 -> base_lr over 'warmup_epochs'
        * Then constant LR for the remaining epochs.
    """

    optimizer = Adam(
        model_params,
        lr=base_lr,
        betas=betas,
        weight_decay=weight_decay
    )

    # Define a Lambda for linear warmup, then constant
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Lightning expects a scheduler dict with additional info
    scheduler_dict = {
        "scheduler": scheduler,
        "interval": "epoch",
        "frequency": 1,
        "name": "linear_warmup_scheduler",
    }

    return optimizer, scheduler_dict


def create_adam_cosine_warmup(
    model_params,
    base_lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.0001,
    warmup_epochs=3,
    total_epochs=100
):
    """
    Returns:
        optimizer, scheduler_dict
    - Optimizer: Adam with base_lr, betas, weight_decay
    - Scheduler: 
        * Linear warmup from 0 -> base_lr over 'warmup_epochs'
        * Then cosine decay from epoch warmup_epochs -> total_epochs
    """

    optimizer = Adam(
        model_params,
        lr=base_lr,
        betas=betas,
        weight_decay=weight_decay
    )

    def lr_lambda(epoch):
        # Linear warmup
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # Cosine decay from warmup_epochs to total_epochs
            progress = float(epoch - warmup_epochs) / float(
                max(1, total_epochs - warmup_epochs)
            )
            # Cosine goes from 1.0 -> 0.0
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    scheduler_dict = {
        "scheduler": scheduler,
        "interval": "epoch",
        "frequency": 1,
        "name": "cosine_decay_scheduler",
    }

    return optimizer, scheduler_dict
