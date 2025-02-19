import os
import torch
import pytorch_lightning as pl
import argparse
from data.data_module import CIFAR10DataModule
from models.recurrent_state_vit import LitRecurrentVisionTransformerWithState
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.save_results import SaveJSONCallback

def load_middle_block_weights(model, checkpoint_path, block_index=5):
    """
    Loads the parameters from the middle block (block_index) of a standard
    Vision Transformer checkpoint into the recurrent block of the provided model.
    
    Args:
        model: Instance of LitRecurrentVisionTransformerWithState.
        checkpoint_path (str): Path to the standard model's .pth checkpoint.
        block_index (int): Zero-indexed block number to load (for block 6 use index 5).
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Assume the standard model saved its transformer blocks under "blocks.<index>.<...>"
    block_prefix = f"blocks.{block_index}."
    middle_block_params = {
        key[len(block_prefix):]: value
        for key, value in state_dict.items() if key.startswith(block_prefix)
    }
    
    print("Extracted the following parameters from the middle block:")
    for k in sorted(middle_block_params.keys()):
        print("  ", k)
    
    # Get the recurrent block from the recurrent model instance.
    recurrent_layer = model.model.recurrent_layer
    recurrent_state = recurrent_layer.state_dict()
    
    # Prepare an updated state dict, mapping keys from the middle block to the recurrent layer.
    updated_state = {}
    for key in recurrent_state.keys():
        if key in middle_block_params:
            updated_state[key] = middle_block_params[key]
            print(f"Mapping parameter '{key}' from checkpoint.")
        else:
            print(f"Parameter '{key}' not found in middle block; using default value.")
    
    # Load the updated parameters (using strict=False in case of minor mismatches)
    recurrent_layer.load_state_dict(updated_state, strict=False)
    print("Recurrent layer initialized with middle block parameters.")

def main(args):
    # Create the data module.
    dm = CIFAR10DataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Instantiate the recurrent model.
    model = LitRecurrentVisionTransformerWithState(
        lr=args.lr,
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        embed_dim=args.embed_dim,
        num_steps=args.depth,  # 'depth' here indicates the number of recurrent steps.
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
    )
    
    # Path to the checkpoint containing the standard model's weights.
    checkpoint_path = os.path.join(
        "results", 
        "vit_swapped_ed256_d12_heads8_lr0.0001_bs512_ep250_wd0.01_20250217_191444_si0.25_ss3.pth"
    )
    
    # Load the middle block (block 6, i.e. index 5) parameters into the recurrent block.
    load_middle_block_weights(model, checkpoint_path, block_index=5)
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [
        SaveJSONCallback(),  # This callback saves training metrics (e.g., accuracies) to a JSON file.
        lr_monitor,
    ]

    # Setup the trainer for 50 epochs.
    if torch.cuda.is_available():
        accelerator = "gpu"
        precision = 16
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        precision = 32
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
    
    # Train and test the model.
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
    
    # Save the final model state (if desired).
    final_model_path = os.path.join("results", "recurrent_initialized_from_block.pth")
    torch.save(model.model.state_dict(), final_model_path)
    print(f"\nFinal model state saved to {final_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12, 
                        help="Number of recurrent steps (use same as original depth)")
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    args = parser.parse_args()
    main(args)
