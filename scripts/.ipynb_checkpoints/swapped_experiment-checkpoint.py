#!/usr/bin/env python3
import os
import subprocess

# Determine the project root directory.
# Assuming swapped_experiment.py is inside scripts/ and your project root is one level up.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define the swap strategies and swap intervals to test.
strategies = [1, 2, 3, 4]
swap_intervals = [10, 5, 2, 1]

# Additional common arguments for train.py.
common_args = (
    "--data_dir ./data "         # Data directory relative to project root.
    "--batch_size 512 "
    "--epochs 250 "
    "--lr 1e-4 "
    "--embed_dim 256 "
    "--depth 12 "
    "--num_heads 8 "
    "--hidden_size 1024 "
    "--dropout 0.1 "
    "--val_split 0.1 "
    "--test "                    # Enable test mode.
    "--weight_decay 0.05 "
    "--model_type vit_swapped "  # Specify the swapped model.
)

# Set the relative path to the training script (relative to project root).
train_script_path = "scripts/train.py"

# Set up the environment variables to ensure PYTHONPATH includes the project root.
env = os.environ.copy()
env["PYTHONPATH"] = project_root

for strategy in strategies:
    for interval in swap_intervals:
        # Construct the full command.
        command = (
            f"python {train_script_path} {common_args}"
            f"--swap_strategy {strategy} --swap_interval {interval}"
        )
        print(f"\nRunning experiment with swap_strategy={strategy} and swap_interval={interval}")
        print(f"Command: {command}")
        # Run the command with the working directory set to the project root.
        subprocess.run(command, shell=True, check=True, cwd=project_root, env=env)
