import torch
import pytorch_lightning as pl
import argparse
from utils.save_results import SaveJSONCallback
from data.data_module import CIFAR10DataModule
from models.vision_transformer import LitVisionTransformer, PrintMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint



import argparse
import json
import os
import torch
import pytorch_lightning as pl
from data.data_module import CIFAR10DataModule
from models.vision_transformer import LitVisionTransformer

def main(args):
    dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    model = LitVisionTransformer(
        lr=args.lr,
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        embed_dim=args.embed_dim,
        depth=args.depth,
        dropout=args.dropout,
    )

    # Create callbacks
    save_json_callback = SaveJSONCallback()
    checkpoint_callback = ModelCheckpoint(
        dirpath="results/checkpoints",
        filename="best_model-{epoch}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )

    # Pick accelerator
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[
            save_json_callback,
            checkpoint_callback,
            PrintMetricsCallback()
        ]
    )

    # Fit (runs train and val)
    trainer.fit(model, dm)

    # Run test if requested
    if args.test:
        trainer.test(model, datamodule=dm)

    # Save final model weights (already handled by SaveJSONCallback for metrics)
    final_model_path = os.path.join("results", "model_weights.pth")
    torch.save(model.model.state_dict(), final_model_path)
    print(f"\nModel parameters saved to {final_model_path}")

    # No need for separate log_data - all metrics in results.json
    print("\nAll training metrics saved to results.json via SaveJSONCallback")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=256) # x 4 = hidden size for mlp
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--test", action="store_true", help="Run test after training.")
    args = parser.parse_args()
    main(args)
