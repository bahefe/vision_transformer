import time
import json
import pytorch_lightning as pl

class SaveJSONCallback(pl.Callback):
    def __init__(self, save_path="results.json"):
        super().__init__()
        self.save_path = save_path
        self.epoch_data = []

    def on_train_epoch_start(self, trainer, pl_module):
        # Record time at the beginning of each epoch
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        # Calculate how long this epoch took
        elapsed = time.time() - self.epoch_start_time
        epoch_mins = elapsed / 60.0

        # Current epoch index
        current_epoch = trainer.current_epoch

        # Get the train accuracy that was logged (if any)
        train_acc = trainer.callback_metrics.get("train_acc")
        if train_acc is not None:
            train_acc = float(train_acc) * 100.0  # Convert from 0-1 to percentage
        else:
            train_acc = None

        # Store these metrics in a list
        self.epoch_data.append({
            "epoch": current_epoch,
            "train_acc": train_acc,
            "time_minutes": round(epoch_mins, 3),
        })

    def on_test_end(self, trainer, pl_module):
        """
        Called once after all test batches are processed.
        We'll dump the epoch data plus final test accuracy to a JSON file.
        """
        test_acc = trainer.callback_metrics.get("test_acc")
        if test_acc is not None:
            test_acc = float(test_acc) * 100.0

        # Wrap all results in a dict
        results = {
            "epochs": self.epoch_data,
            "final_test_acc": test_acc
        }

        # Write to JSON
        with open(self.save_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"[SaveJSONCallback] Results saved to {self.save_path}")
