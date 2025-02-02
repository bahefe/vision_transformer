import torch
import pytorch_lightning as pl
import argparse
from utils.save_results import SaveJSONCallback
from data.data_module import CIFAR10DataModule
from models.vision_transformer import LitVisionTransformer
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import json
import os
from pytorch_lightning.callbacks import LearningRateMonitor
import random
import torch.nn as nn
from models.vision_transformer import LitVisionTransformer
from models.recurrent_vit import LitRecurrentVisionTransformer



class SwapEncoderBlocksCallback(pl.Callback):
    def __init__(self, swap_interval=20):  # Modified
        super().__init__()
        self.swap_interval = swap_interval  # Modified

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        # Modified condition below
        if self.swap_interval > 0 and current_epoch > 0 and current_epoch % self.swap_interval == 0:
            if hasattr(pl_module.model, 'blocks') and isinstance(pl_module.model.blocks, nn.ModuleList):
                blocks = pl_module.model.blocks
                if len(blocks) >= 2:
                    idx1, idx2 = random.sample(range(len(blocks)), 2)
                    blocks[idx1], blocks[idx2] = blocks[idx2], blocks[idx1]
                    print(f"Epoch {current_epoch}: Swapped encoder blocks {idx1} and {idx2} (interval={self.swap_interval})")

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
        )
    elif args.model_type == "standard" or args.model_type == "vit_swapped":
        model = LitVisionTransformer(
            lr=args.lr,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            embed_dim=args.embed_dim,
            depth=args.depth,
            dropout=args.dropout,
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
        # Modified line below to pass swap_interval
        callbacks.append(SwapEncoderBlocksCallback(swap_interval=args.swap_interval))

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

    final_model_path = os.path.join("results", "model_weights.pth")
    torch.save(model.model.state_dict(), final_model_path)
    print(f"\nModel parameters saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # New argument added here
    parser.add_argument("--swap_interval", type=int, default=20,
                       help="Swap encoder blocks every N epochs (for 'vit_swapped')")
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
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay (L2 regularization factor)")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args)