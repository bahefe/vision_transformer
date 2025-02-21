# Filename: train_latent_vit.py

import os
import argparse
import torch
import pytorch_lightning as pl

# 1) Import your data module and model
from data.data_module import CIFAR10DataModule
from models.latent_space_vit import LitLatentSpaceVisionTransformer

# ----------------------------------------------------------------------------
# (A) HELPER FUNCTIONS TO LOAD BLOCKS FROM A PRETRAINED VIT CHECKPOINT
# ----------------------------------------------------------------------------
def load_vit_block_into_module(module, state_dict, block_index):
    """
    Loads parameters from a single ViT block (e.g. 'blocks.<block_index>.*')
    into the given `module` (which should be a TransformerEncoderBlock).
    We use strict=False in case of minor mismatches in shapes or missing keys.
    """
    block_prefix = f"blocks.{block_index}."

    # Filter for keys belonging to blocks.<block_index>
    block_params = {
        key[len(block_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(block_prefix)
    }

    print(f"Loading block {block_index} into {module.__class__.__name__}...")
    if not block_params:
        print(f"  No params found for block {block_index}. Check your checkpoint keys.")
        return

    for k in sorted(block_params.keys()):
        print("   ", k)

    module.load_state_dict(block_params, strict=False)
    print("  [Done]\n")


def load_latent_space_vit_weights(model, checkpoint_path, 
                                  first_block_idx=0, 
                                  mid_block_idx=5, 
                                  last_block_idx=11):
    """
    Given a pretrained ViT checkpoint, load:
      - block `first_block_idx` -> model.initial_block
      - block `mid_block_idx`   -> model.recurrent_block
      - block `last_block_idx`  -> model.final_block
    in the LatentSpaceVisionTransformer.
    """
    print(f"Loading ViT checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # 1) initial_block
    load_vit_block_into_module(model.initial_block, state_dict, first_block_idx)
    # 2) recurrent_block
    load_vit_block_into_module(model.recurrent_block, state_dict, mid_block_idx)
    # 3) final_block
    load_vit_block_into_module(model.final_block, state_dict, last_block_idx)

    print("Finished loading blocks into LatentSpaceVisionTransformer.\n")

# ----------------------------------------------------------------------------
# (B) MAIN SCRIPT
# ----------------------------------------------------------------------------
def main(args):
    # 1) Instantiate the data module (already defined in data.data_module)
    dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )

    # 2) Instantiate your LatentSpaceVisionTransformer Lightning module
    model = LitLatentSpaceVisionTransformer(
        model_type="latent_space",
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

    # 3) Optionally load blocks from the standard ViT checkpoint
    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        load_latent_space_vit_weights(
            model=model.model,
            checkpoint_path=args.checkpoint_path,
            first_block_idx=args.first_block_idx,
            mid_block_idx=args.mid_block_idx,
            last_block_idx=args.last_block_idx
        )
    else:
        print("No valid checkpoint path provided. Training from scratch.")

    # 4) Setup PyTorch Lightning Trainer
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
        precision = 16
    else:
        accelerator = "cpu"
        devices = 1
        precision = 32

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,  # or you can rely on your model's internally set epochs
        accelerator=accelerator,
        devices=devices,
        precision=precision
    )

    # 5) Train and Test
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

    # 6) Save the final model
    final_model_path = os.path.join("results", "latent_space_vit_finetuned.pth")
    torch.save(model.model.state_dict(), final_model_path)
    print(f"\nFinal model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and training arguments
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=10, help="Override total epochs if needed")

    # Checkpoint + block indices
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to the standard ViT .pth checkpoint file")
    parser.add_argument("--first_block_idx", type=int, default=0,
                        help="Which block to map to initial_block")
    parser.add_argument("--mid_block_idx", type=int, default=5,
                        help="Which block to map to recurrent_block")
    parser.add_argument("--last_block_idx", type=int, default=11,
                        help="Which block to map to final_block")

    args = parser.parse_args()
    main(args)
