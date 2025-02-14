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
from models.recurrent_state_vit import LitRecurrentVisionTransformerWithState
from models.latent_space_vit import LitLatentSpaceVisionTransformer  # New import

class SwapEncoderBlocksCallback(pl.Callback):
    def __init__(self, swap_interval=0.25, strategy=1, log_file="swap_log.json"):
        super().__init__()
        self.swap_interval = swap_interval
        self.strategy = strategy
        self.log_file = log_file
        self.swap_events = []
        self.next_swap_point = 0.0  # Track progress for next swap
        self.current_epoch = 0
        self.num_training_batches = 0

    def on_train_epoch_start(self, trainer, pl_module):
        # Update current epoch and number of training batches
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

    # Strategy methods return log data instead of printing
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

    def on_train_end(self, trainer, pl_module):
        # Save swap events to log file
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

    if args.model_type == "recurrent_state":
        model = LitRecurrentVisionTransformerWithState(
            lr=args.lr,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            embed_dim=args.embed_dim,
            num_steps=args.depth,  # using 'depth' as the number of recurrent steps
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
        )
    elif args.model_type == "recurrent":
        model = LitRecurrentVisionTransformer(
            lr=args.lr,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            embed_dim=args.embed_dim,
            num_steps=args.depth,  # using 'depth' as the number of recurrent steps
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
        )
    elif args.model_type == "latent_space":
        model = LitLatentSpaceVisionTransformer(
            lr=args.lr,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            embed_dim=args.embed_dim,
            depth_recurrent=args.depth,  # using 'depth' as the number of recurrent iterations
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
    parser.add_argument("--swap_interval", type=float, default=0.25,
                        help="Swap interval in epochs (can be fractional, e.g., 0.25 for every quarter epoch)")
    parser.add_argument("--swap_strategy", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Swapping strategy. 1=Neighbor swap (all), 2=Full permutation (all), 3=Neighbor swap (middle only), 4=Permutation (middle only).")
    parser.add_argument("--val_split", type=float, default=0.1)
    # Updated model_type choices to include 'latent_space'
    parser.add_argument("--model_type", type=str, default="standard",
                        choices=["standard", "recurrent", "recurrent_state", "vit_swapped", "latent_space"])
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
