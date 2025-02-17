# train_latent_reasoning.py

import torch
import pytorch_lightning as pl
import argparse
import os
from datetime import datetime

# Import your CIFAR10DataModule and utility callbacks
from data.data_module import CIFAR10DataModule
from utils.save_results import SaveJSONCallback
from pytorch_lightning.callbacks import LearningRateMonitor

# Import the new model class
# Make sure you've placed the model code into this file or in "models/recurrent_latent_reasoning_vit.py"
# and adjust the import path accordingly.
from models.latent_reasoning import LitLatentReasoningVisionTransformer

def main(args):
    # 1) Data Module
    dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    # 2) Create the Model
    #    We map command line arguments to model constructor args as needed.
    model = LitLatentReasoningVisionTransformer(
        lr=args.lr,
        weight_decay=args.weight_decay,
        # Pass in all the custom parameters for your Recurrent ViT:
        img_size=32,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10,
        embed_dim=args.embed_dim,
        l_P=args.l_prelude,
        l_R=args.l_recurrent,
        l_C=args.l_coda,
        num_heads=args.num_heads,
        mlp_hidden_dim=args.hidden_size,
        adapter_injection=args.adapter_injection,
        recurrent_hidden_dim=args.recurrent_hidden_size,
        mean_recurrence=args.mean_recurrence,
        mean_backprop_depth=args.mean_backprop_depth,
        dropout=args.dropout,
    )

    # 3) Create callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [
        SaveJSONCallback(),
        lr_monitor
    ]

    # 4) Determine accelerator & precision
    if torch.backends.mps.is_available():
        accelerator = "mps"
        precision = 32
    elif torch.cuda.is_available():
        accelerator = "gpu"
        precision = "16-mixed"
    else:
        accelerator = "cpu"
        precision = 32

    # 5) Setup Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        devices=1,
        accelerator=accelerator,
        precision=precision,
    )

    # 6) Fit
    trainer.fit(model, dm)

    # 7) Optional test
    if args.test:
        trainer.test(model, datamodule=dm)

    # 8) Save final model weights
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_filename = (
        f"latent_reasoning_"
        f"ed{args.embed_dim}_"
        f"heads{args.num_heads}_"
        f"mp{args.mean_recurrence}_"
        f"lr{args.lr}_"
        f"bs{args.batch_size}_"
        f"ep{args.epochs}_"
        f"wd{args.weight_decay}_"
        f"{timestamp}.pth"
    )
    final_model_path = os.path.join("results", final_model_filename)
    torch.save(model.model.state_dict(), final_model_path)
    print(f"\nModel parameters saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # -- Basic Training Hyperparams --
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test", action="store_true")

    # -- ViT-related Hyperparams --
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=1024)

    # -- Recurrent Core Hyperparams --
    parser.add_argument("--l_prelude", type=int, default=1,
                        help="# of sandwich blocks in the prelude")
    parser.add_argument("--l_recurrent", type=int, default=4,
                        help="# of sandwich blocks in each recurrent iteration")
    parser.add_argument("--l_coda", type=int, default=1,
                        help="# of sandwich blocks in the coda")
    parser.add_argument("--recurrent_hidden_size", type=int, default=2048,
                        help="hidden size used inside the recurrent block MLP")
    parser.add_argument("--adapter_injection", action="store_true",
                        help="Use concat+linear injection instead of add injection")
    parser.add_argument("--mean_recurrence", type=int, default=10,
                        help="Total # of recurrent iterations per forward pass")
    parser.add_argument("--mean_backprop_depth", type=int, default=4,
                        help="# of those iterations done with gradient (truncated backprop)")

    args = parser.parse_args()
    main(args)
