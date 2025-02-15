import time
import json
import torch
import pytorch_lightning as pl
import os

class SaveJSONCallback(pl.Callback):
    def __init__(self, output_dir="results", base_filename=None):
        super().__init__()
        self.output_dir = output_dir
        self.base_filename = base_filename
        self.epoch_data = []
        self.save_path = None
        self.epoch_start_time = None

    def on_fit_start(self, trainer, pl_module):
        os.makedirs(self.output_dir, exist_ok=True)
        hparams = pl_module.hparams
        model_type = hparams.get("model_type", "model")
        # Try to retrieve batch_size; if not present, set to "unknown"
        batch_size = getattr(hparams, "batch_size", "unknown")
        base = (
            f"{model_type}_"
            f"ed{hparams.embed_dim}_"
            f"d{getattr(hparams, 'depth', getattr(hparams, 'num_steps', getattr(hparams, 'depth_recurrent', 'unknown')))}_"
            f"heads{hparams.num_heads}_"
            f"hs{hparams.hidden_size}_"
            f"bs{batch_size}_"
            f"ep{hparams.epochs}_"
            f"wd{hparams.weight_decay}"
        )
        if model_type == "vit_swapped":
            base += f"_si{hparams.swap_interval}_ss{hparams.swap_strategy}"
        base += f"_{time.strftime('%Y%m%d-%H%M%S')}"
        self.base_filename = base
        self.save_path = os.path.join(self.output_dir, f"{self.base_filename}.json")



    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start_time
        epoch_mins = elapsed / 60.0
        current_epoch = trainer.current_epoch

        # Get training metrics
        train_loss = trainer.callback_metrics.get("train_loss")
        train_acc = trainer.callback_metrics.get("train_acc")
        train_acc = float(train_acc) * 100.0 if train_acc else None

        # Create new epoch entry with training data
        self.epoch_data.append({
            "epoch": current_epoch,
            "train_loss": float(train_loss) if train_loss else None,
            "train_acc": train_acc,
            "time_minutes": round(epoch_mins, 3),
            "val_loss": None,  # Initialize validation metrics
            "val_acc": None,
        })

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get validation metrics
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        
        if val_loss is not None and val_acc is not None and self.epoch_data:
            # Update latest epoch entry with validation metrics
            self.epoch_data[-1]["val_loss"] = float(val_loss)
            self.epoch_data[-1]["val_acc"] = float(val_acc) * 100.0

    def on_test_end(self, trainer, pl_module):
        # Get final test metrics
        test_loss = trainer.callback_metrics.get("test_loss")
        test_acc = trainer.callback_metrics.get("test_acc")
        test_acc = float(test_acc) * 100.0 if test_acc else None

        # Gather swap events from any callback that has a 'swap_events' attribute.
        swap_events = []
        for cb in trainer.callbacks:
            if hasattr(cb, "swap_events"):
                swap_events.extend(cb.swap_events)
        
        # Prepare final results
        results = {
            "hyperparameters": dict(pl_module.hparams),  # Save all hyperparameters
            "epochs": self.epoch_data,
            "final_test_loss": float(test_loss) if test_loss else None,
            "final_test_acc": test_acc,
            "swap_events": swap_events,  # Add swap events (if any)
        }

        with open(self.save_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[SaveJSONCallback] Results saved to {self.save_path}")
