import torch
import pytorch_lightning as pl
import argparse

from data.data_module import CIFAR10DataModule
from models.vision_transformer import LitVisionTransformer, PrintMetricsCallback

def main(args):
    # Basic data module (no val)
    dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size)

    # Minimal model config
    model = LitVisionTransformer(
        lr=args.lr,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1
    )

    # Choose device
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Basic trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[PrintMetricsCallback()],
    )

    # Train
    trainer.fit(model, dm)

    # Test (optional)
    if args.test:
        trainer.test(model, dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args)
