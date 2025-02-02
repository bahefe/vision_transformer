import subprocess

# Define the list of strategies and swap intervals to try.
strategies = [1, 2, 3, 4]
swap_intervals = [10, 5, 2, 1]

# Optional: Specify additional arguments you want to pass to the training script.
# For example, you might set epochs, learning rate, etc.
common_args = (
    "--model_type vit_swapped "  # Ensure using the swapped model.
    # You can add other common arguments here if desired:
    # "--epochs 10 --lr 1e-4 --embed_dim 256 --depth 6 --patch_size 4 --num_heads 8 --batch_size 512 "
)

for strategy in strategies:
    for interval in swap_intervals:
        # Construct the command line arguments.
        command = (
            f"python train.py {common_args}"
            f"--swap_strategy {strategy} "
            f"--swap_interval {interval}"
        )
        print(f"\nRunning experiment with swap_strategy={strategy} and swap_interval={interval}")
        print(f"Command: {command}")
        
        # Run the training script. Using shell=True for convenience here.
        subprocess.run(command, shell=True, check=True)
