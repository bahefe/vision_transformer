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

    # pick accelerator
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
    )

    # Fit (runs train and val)
    trainer.fit(model, dm)

    # Log results to a file
    val_acc = trainer.callback_metrics.get("val_acc")
    log_data = {
        "patch_size": args.patch_size,
        "num_heads": args.num_heads,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "val_acc": float(val_acc) * 100 if val_acc is not None else None
    }

    # Create logs directory if it doesn't exist
    os.makedirs("results/logs", exist_ok=True)
    log_file = os.path.join("results/logs", f"log_patch{args.patch_size}_heads{args.num_heads}_batch{args.batch_size}.json")

    # Save log to a file
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)
    print(f"Results logged to {log_file}")

    # Optional test
    if args.test:
        trainer.test(model, datamodule=dm)
        test_acc = trainer.callback_metrics.get("test_acc")
        print(f"Final test_acc: {float(test_acc) * 100:.2f}%" if test_acc is not None else "No test accuracy logged.")

    # Inside main() function after trainer.fit(...):

    # Save full model weights
    final_model_path = os.path.join("results", "model_weights.pth")
    torch.save(model.model.state_dict(), final_model_path)
    print(f"\nModel parameters saved to {final_model_path}")
    
    # Optional: Save full Lightning checkpoint (includes optimizer state)
    checkpoint_path = os.path.join("results", "full_checkpoint.ckpt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"Full checkpoint saved to {checkpoint_path}")

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
