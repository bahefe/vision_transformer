import torch
import pytorch_lightning as pl
import argparse

# Import the DataModule & Model
from data.data_module import CIFAR10DataModule
from models.vision_transformer import LitVisionTransformer, PrintMetricsCallback
# Make sure this callback is defined in your utils/save_results.py
from utils.save_results import SaveLayerWeightsCallback

def main(args):
    # 1. Instantiate data module
    dm = CIFAR10DataModule(data_dir=args.data_dir, batch_size=args.batch_size)

    # 2. Instantiate model (here is where you can experiment with depth, etc.)
    model = LitVisionTransformer(
        lr=args.lr,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        depth=args.depth,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        optimizer_type=args.optimizer_type,  # <--- pass optimizer type to the model
        max_epochs=args.epochs,             # <--- so schedulers can reference total epochs if needed
    )

    # 3. Pick an accelerator
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # 4. Create the trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[PrintMetricsCallback(), SaveLayerWeightsCallback()],
    )

    # 5. Fit / Train
    trainer.fit(model, dm)

    # 6. Test (optional)
    if args.test:
        trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Where CIFAR10 is stored")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--depth", type=int, default=6, help="Number of ViT encoder blocks")

    # NEW: Add --optimizer_type
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adam_linear_warmup",
        help="Which optimizer to use. e.g. 'adam_linear_warmup', 'adam_cosine_warmup', or 'default'"
    )

    parser.add_argument("--test", action="store_true", help="Run test after training")

    args = parser.parse_args()
    main(args)
