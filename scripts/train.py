import torch
import pytorch_lightning as pl
import argparse
from utils.save_results import SaveJSONCallback
from data.data_module import CIFAR10DataModule
from models.vision_transformer import LitVisionTransformer
import json
from pytorch_lightning.callbacks import LearningRateMonitor
import random
import torch.nn as nn
from models.vision_transformer import LitVisionTransformer
from models.recurrent_state_vit import LitRecurrentVisionTransformerWithState
from models.latent_space_vit import LitLatentSpaceVisionTransformer  
from utils.swap_helpers import SwapEncoderBlocksCallback
import os
from datetime import datetime


def main(args):
    dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    if args.model_type == "recurrent_state":
        model = LitRecurrentVisionTransformerWithState(
            model_type=args.model_type,
            lr=args.lr,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            embed_dim=args.embed_dim,
            num_steps=args.depth,  # using 'depth' as the number of recurrent steps
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,  # Added
            epochs=args.epochs,          # Added
        )

    
    elif args.model_type == "latent_space":
        model = LitLatentSpaceVisionTransformer(
            model_type=args.model_type,
            lr=args.lr,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            embed_dim=args.embed_dim,
            depth_recurrent=args.depth,  # using 'depth' as the number of recurrent iterations
            hidden_size=args.hidden_size,
            recurrent_hidden_size=args.recurrent_hidden_size,  # New parameter passed here
            dropout=args.dropout,
            weight_decay=args.weight_decay,
        )
    elif args.model_type in ["standard", "vit_swapped"]:
        model = LitVisionTransformer(
            model_type=args.model_type,
            lr=args.lr,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            embed_dim=args.embed_dim,
            depth=args.depth,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,  # Added
            epochs=args.epochs,
            swap_interval=args.swap_interval if args.model_type == "vit_swapped" else None,
            swap_strategy=args.swap_strategy if args.model_type == "vit_swapped" else None,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Create a LearningRateMonitor callback to log the learning rate.
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Create a list of callbacks. The ModelCheckpoint callback has been removed.
    callbacks = [
        SaveJSONCallback(),
        lr_monitor
    ]

    if args.model_type == "vit_swapped":
        callbacks.append(
            SwapEncoderBlocksCallback(
                swap_interval=args.swap_interval, 
                strategy=args.swap_strategy
            )
        )

    # Determine the accelerator and precision based on hardware availability.
    if torch.backends.mps.is_available():
        accelerator = "mps"
        precision = 32
    elif torch.cuda.is_available():
        accelerator = "gpu"
        precision = "16-mixed"   # Use 16-mixed for GPU
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_filename = (
        f"{args.model_type}_"
        f"ed{args.embed_dim}_"
        f"d{args.depth}_"
        f"hs{args.hidden_size}_"
        f"heads{args.num_heads}_"
        f"lr{args.lr}_"
        f"bs{args.batch_size}_"
        f"ep{args.epochs}_"
        f"wd{args.weight_decay}_"
        f"{timestamp}"
    )
    
    
    if args.model_type == "vit_swapped":
        final_model_filename += f"_si{args.swap_interval}_ss{args.swap_strategy}"
    final_model_filename += ".pth"

    final_model_path = os.path.join("results", final_model_filename)
    
    torch.save(model.model.state_dict(), final_model_path)
    print(f"\nModel parameters saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--swap_interval", type=float, default=0.25,
                        help="Swap interval in epochs (can be fractional, e.g., 0.25 for every quarter epoch)")
    parser.add_argument("--swap_strategy", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Swapping strategy. 1=Neighbor swap (all), 2=Full permutation (all), 3=Neighbor swap (middle only), 4=Permutation (middle only).")
    parser.add_argument("--model_type", type=str, default="standard",
                        choices=["standard", "recurrent_state", "vit_swapped", "latent_space"])
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--recurrent_hidden_size", type=int, default=14800,
                        help="Hidden size for the recurrent block (inflated MLP dimension)")
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
