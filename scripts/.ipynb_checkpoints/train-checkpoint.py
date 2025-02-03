import torch
import pytorch_lightning as pl
import argparse
from utils.save_results import SaveJSONCallback
from data.data_module import CIFAR10DataModule
from models.vision_transformer import LitVisionTransformer
from pytorch_lightning.callbacks import ModelCheckpoint
import json
import os
from pytorch_lightning.callbacks import LearningRateMonitor
import random
import torch.nn as nn
from models.vision_transformer import LitVisionTransformer
from models.recurrent_vit import LitRecurrentVisionTransformer


class SwapEncoderBlocksCallback(pl.Callback):
    """
    A callback that implements four different swapping/permutation strategies
    for the encoder blocks in a Vision Transformer and saves swap details.

    Strategies:
      1) Strategy 1: Random neighbor swap across all blocks.
         - Logs the two swapped indices.
      2) Strategy 2: Full permutation of all blocks.
         - Logs the new order as a list of indices.
      3) Strategy 3: Random neighbor swap, but only among the middle blocks
         (excluding the first and last).
         - Logs the two swapped indices.
      4) Strategy 4: Full permutation, but only among the middle blocks
         (excluding the first and last).
         - Logs the new order (of the middle block indices).
    """
    def __init__(self, swap_interval=20, strategy=1, log_file="swap_log.json"):
        """
        :param swap_interval: Perform the swap/permutation every 'swap_interval' epochs.
        :param strategy: An integer (1, 2, 3, or 4) specifying which strategy to use.
        :param log_file: File name where swap events will be saved at training end.
        """
        super().__init__()
        self.swap_interval = swap_interval
        self.strategy = strategy
        self.log_file = log_file
        self.swap_events = []  # List to record swap events

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        # Only run if the epoch > 0 and the epoch is a multiple of swap_interval.
        if self.swap_interval > 0 and current_epoch > 0 and current_epoch % self.swap_interval == 0:
            if not hasattr(pl_module.model, 'blocks'):
                return  # No blocks to swap
            blocks = pl_module.model.blocks

            if not isinstance(blocks, nn.ModuleList):
                return  # We expect blocks to be an nn.ModuleList

            n = len(blocks)
            if n < 2:
                return  # Not enough blocks to swap/permute

            # Decide the strategy
            if self.strategy == 1:
                self._strategy_1_swap_all(blocks, current_epoch)
            elif self.strategy == 2:
                self._strategy_2_full_permutation(blocks, current_epoch)
            elif self.strategy == 3:
                self._strategy_3_swap_middle(blocks, current_epoch)
            elif self.strategy == 4:
                self._strategy_4_permute_middle(blocks, current_epoch)
            else:
                print(f"[SwapEncoderBlocksCallback] Unknown strategy: {self.strategy}")

    def _strategy_1_swap_all(self, blocks, current_epoch):
        """
        Strategy 1: Pick a random index i among [0 .. n-1],
        and swap it with (i+1) % n.
        """
        n = len(blocks)
        i = random.randint(0, n - 1)
        j = (i + 1) % n
        blocks[i], blocks[j] = blocks[j], blocks[i]
        msg = f"Epoch {current_epoch}: Strategy 1 swapped blocks {i} and {j}"
        print(msg)
        self.swap_events.append({
            "epoch": current_epoch,
            "strategy": 1,
            "swapped": [i, j]
        })

    def _strategy_2_full_permutation(self, blocks, current_epoch):
        """
        Strategy 2: Randomly permute ALL blocks at once.
        """
        n = len(blocks)
        indices = list(range(n))
        random.shuffle(indices)
        # Reorder blocks in place according to the shuffled indices
        permuted = [blocks[idx] for idx in indices]
        for i in range(n):
            blocks[i] = permuted[i]
        msg = f"Epoch {current_epoch}: Strategy 2 permuted all {n} blocks. New order: {indices}"
        print(msg)
        self.swap_events.append({
            "epoch": current_epoch,
            "strategy": 2,
            "new_order": indices
        })

    def _strategy_3_swap_middle(self, blocks, current_epoch):
        """
        Strategy 3: Same as strategy 1, but only for the "middle" blocks:
        i in [1 .. n-2]. If i == n-2, we wrap around to 1.
        The first and last blocks (index 0 and n-1) stay fixed.
        """
        n = len(blocks)
        if n <= 2:
            return  # There's no middle to swap if n <= 2
        i = random.randint(1, n - 2)
        # The next index would be i+1, but if i == n-2, we wrap to 1
        if i == (n - 2):
            j = 1
        else:
            j = i + 1
        blocks[i], blocks[j] = blocks[j], blocks[i]
        msg = f"Epoch {current_epoch}: Strategy 3 swapped middle blocks {i} and {j}"
        print(msg)
        self.swap_events.append({
            "epoch": current_epoch,
            "strategy": 3,
            "swapped": [i, j]
        })

    def _strategy_4_permute_middle(self, blocks, current_epoch):
        """
        Strategy 4: Randomly permute only the middle blocks [1 .. n-2],
        keeping the first block (index 0) and last block (index n-1) in place.
        """
        n = len(blocks)
        if n <= 2:
            return  # No middle blocks to permute
        middle_indices = list(range(1, n - 1))
        random.shuffle(middle_indices)
        # Extract the middle blocks
        middle_blocks = [blocks[idx] for idx in range(1, n - 1)]
        # Now reorder them according to middle_indices
        permuted = [None] * len(middle_blocks)
        for k, idx in enumerate(middle_indices):
            permuted[k] = blocks[idx]
        # Put them back into blocks
        for k, idx in enumerate(range(1, n - 1)):
            blocks[idx] = permuted[k]
        msg = f"Epoch {current_epoch}: Strategy 4 permuted middle blocks (1..{n-2}). New order: {middle_indices}"
        print(msg)
        self.swap_events.append({
            "epoch": current_epoch,
            "strategy": 4,
            "new_order": middle_indices
        })

    def on_train_end(self, trainer, pl_module):
        # Save the swap events to a file in the results directory.
        log_dir = "results"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, self.log_file)
        with open(log_path, "w") as f:
            json.dump(self.swap_events, f, indent=4)
        print(f"Swap events saved to {log_path}")


def main(args):
    dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    if args.model_type == "recurrent":
        model = LitRecurrentVisionTransformer(
            lr=args.lr,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            embed_dim=args.embed_dim,
            num_steps=args.depth,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
        )
    elif args.model_type in ["standard", "vit_swapped"]:
        model = LitVisionTransformer(
            lr=args.lr,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            embed_dim=args.embed_dim,
            depth=args.depth,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    callbacks = [
        SaveJSONCallback(),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="best_model-{epoch}-{val_acc:.2f}",
            monitor="val_acc",
            mode="max",
            save_top_k=1
        ),
        lr_monitor
    ]

    if args.model_type == "vit_swapped":
        callbacks.append(
            SwapEncoderBlocksCallback(
                swap_interval=args.swap_interval, 
                strategy=args.swap_strategy
            )
        )

    if torch.backends.mps.is_available():
        accelerator = "mps"
        precision = 32
    elif torch.cuda.is_available():
        accelerator = "gpu"
        precision = 16
    else:
        accelerator = "cpu"
        precision = 32

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        devices=1,
        accelerator=accelerator,
        precision=precision,
    )
        
    trainer.fit(model, dm)

    if args.test:
        trainer.test(model, datamodule=dm)

    # Build a descriptive filename
    final_model_filename = (
        f"{args.model_type}_"
        f"ed{args.embed_dim}_"
        f"d{args.depth}_"
        f"heads{args.num_heads}_"
        f"lr{args.lr}_"
        f"bs{args.batch_size}_"
        f"ep{args.epochs}_"
        f"wd{args.weight_decay}"
    )
    # Append swap_interval and swap_strategy if using the swapped model
    if args.model_type == "vit_swapped":
        final_model_filename += f"_si{args.swap_interval}_ss{args.swap_strategy}"
    final_model_filename += ".pth"

    final_model_path = os.path.join("results", final_model_filename)
    
    # Save just the underlying nn.Module's state_dict
    torch.save(model.model.state_dict(), final_model_path)
    print(f"\nModel parameters saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--swap_interval", type=int, default=20,
                        help="Swap encoder blocks every N epochs (for 'vit_swapped')")
    parser.add_argument("--swap_strategy", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Swapping strategy. 1=Neighbor swap (all), 2=Full permutation (all), 3=Neighbor swap (middle only), 4=Permutation (middle only).")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--model_type", type=str, default="standard",
                        choices=["standard", "recurrent", "vit_swapped"])
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay (L2 regularization factor)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args)
