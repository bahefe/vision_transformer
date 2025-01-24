import subprocess
import os
import json

def main():
    # Hyperparameter search space
    patch_sizes = [2, 4, 8]
    head_list = [2, 4, 8]
    batch_sizes = [64, 128, 256]
    results_folder = "results/hyperparam_tuning"
    logs_folder = "results/logs"  # Folder for logs

    # Create results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)

    results = []  # Store all results in memory

    for patch in patch_sizes:
        for heads in head_list:
            for batch_size in batch_sizes:
                # Command to call the train.py script
                cmd = [
                    "python", "-m", "scripts.train",  # Use -m for module imports
                    "--epochs", "10",
                    "--batch_size", str(batch_size),
                    "--lr", "0.001",
                    "--embed_dim", "256",
                    "--depth", "6",
                    "--dropout", "0.1",
                    "--patch_size", str(patch),
                    "--num_heads", str(heads),
                ]
                print(f"\n[hyperparam_tune] Running: {cmd}")

                # Run the training script
                subprocess.run(cmd)

                # Read the corresponding log file
                log_file = os.path.join(logs_folder, f"log_patch{patch}_heads{heads}_batch{batch_size}.json")
                if os.path.exists(log_file):
                    with open(log_file, "r") as f:
                        log_data = json.load(f)
                        results.append(log_data)
                else:
                    print(f"WARNING: Log file {log_file} not found!")

    # Save all results into a single JSON file
    with open(os.path.join(results_folder, "all_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[hyperparam_tune] Results saved in {results_folder}")

if __name__ == "__main__":
    main()
