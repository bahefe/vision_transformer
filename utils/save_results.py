import torch
import os
import pytorch_lightning as pl

class SaveLayerWeightsCallback(pl.Callback):
    def __init__(self, save_dir="layer_weights", save_freq=1):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_freq = save_freq

    def on_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch % self.save_freq == 0:
            # Example: save each encoder block's parameters
            state_dict = pl_module.model.state_dict()
            torch.save(state_dict, os.path.join(self.save_dir, f"epoch_{current_epoch}.pt"))
            print(f"Saved layer weights for epoch {current_epoch}")
