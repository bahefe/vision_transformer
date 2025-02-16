import os
import json
import random
import torch.nn as nn
import pytorch_lightning as pl

class SwapEncoderBlocksCallback(pl.Callback):
    def __init__(self, swap_interval=0.25, strategy=1):
        super().__init__()
        self.swap_interval = swap_interval
        self.strategy = strategy
        self.swap_events = []
        self.next_swap_point = swap_interval  # Track progress for next swap
        self.current_epoch = 0
        self.num_training_batches = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch = trainer.current_epoch
        self.num_training_batches = trainer.num_training_batches
        if self.num_training_batches == 0:
            self.num_training_batches = 1  # Prevent division by zero

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Calculate current progress (epoch + batch progress)
        current_progress = self.current_epoch + (batch_idx / self.num_training_batches)
        
        # Perform swaps if current progress exceeds the next swap point
        while current_progress >= self.next_swap_point:
            self._perform_swap(pl_module, current_progress, batch_idx)
            self.next_swap_point += self.swap_interval

    def _perform_swap(self, pl_module, current_progress, batch_idx):
        if not hasattr(pl_module.model, 'blocks'):
            return
        blocks = pl_module.model.blocks
        if not isinstance(blocks, nn.ModuleList) or len(blocks) < 2:
            return

        log_data = None
        if self.strategy == 1:
            log_data = self._strategy_1_swap_all(blocks)
        elif self.strategy == 2:
            log_data = self._strategy_2_full_permutation(blocks)
        elif self.strategy == 3:
            log_data = self._strategy_3_swap_middle(blocks)
        elif self.strategy == 4:
            log_data = self._strategy_4_permute_middle(blocks)
        else:
            return

        # Log swap event with progress and batch details
        self.swap_events.append({
            "progress": current_progress,
            "epoch": self.current_epoch,
            "batch_idx": batch_idx,
            "strategy": self.strategy,
            **log_data
        })
        print(f"Swap at {current_progress:.2f} epochs: {log_data}")

    def _strategy_1_swap_all(self, blocks):
        n = len(blocks)
        i = random.randint(0, n-1)
        j = (i + 1) % n
        blocks[i], blocks[j] = blocks[j], blocks[i]
        return {"swapped": [i, j]}

    def _strategy_2_full_permutation(self, blocks):
        n = len(blocks)
        indices = list(range(n))
        random.shuffle(indices)
        permuted = [blocks[i] for i in indices]
        for i in range(n):
            blocks[i] = permuted[i]
        return {"new_order": indices}

    def _strategy_3_swap_middle(self, blocks):
        n = len(blocks)
        if n <= 2:
            return {"swapped": None}
        i = random.randint(1, n-2)
        j = i+1 if i < n-2 else 1
        blocks[i], blocks[j] = blocks[j], blocks[i]
        return {"swapped": [i, j]}

    def _strategy_4_permute_middle(self, blocks):
        n = len(blocks)
        if n <= 2:
            return {"new_order": None}
        middle_indices = list(range(1, n-1))
        random.shuffle(middle_indices)
        permuted = [blocks[i] for i in middle_indices]
        for idx, pos in enumerate(range(1, n-1)):
            blocks[pos] = permuted[idx]
        return {"new_order": middle_indices}

    
