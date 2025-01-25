import time
import json
import pytorch_lightning as pl

class SaveJSONCallback(pl.Callback):
    def __init__(self, save_path="results.json"):
        super().__init__()
        self.save_path = save_path
        self.epoch_data = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self.epoch_start_time
        epoch_mins = elapsed / 60.0
        current_epoch = trainer.current_epoch

        # Get training metrics
        train_acc = trainer.callback_metrics.get("train_acc")
        train_acc = float(train_acc) * 100.0 if train_acc else None

        # Create new epoch entry with training data
        self.epoch_data.append({
            "epoch": current_epoch,
            "train_acc": train_acc,
            "time_minutes": round(epoch_mins, 3),
            "val_acc": None  # Initialize val_acc slot
        })

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get validation accuracy
        val_acc = trainer.callback_metrics.get("val_acc")
        
        if val_acc is not None and self.epoch_data:
            # Update latest epoch entry with validation accuracy
            val_acc_pct = float(val_acc) * 100.0
            self.epoch_data[-1]["val_acc"] = round(val_acc_pct, 2)

    def on_test_end(self, trainer, pl_module):
        # Get final test accuracy
        test_acc = trainer.callback_metrics.get("test_acc")
        test_acc = float(test_acc) * 100.0 if test_acc else None

        # Prepare final results
        results = {
            "epochs": self.epoch_data,
            "final_test_acc": round(test_acc, 2) if test_acc else None
        }

        # Write to JSON
        with open(self.save_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"[SaveJSONCallback] Results saved to {self.save_path}")