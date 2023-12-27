import os
from pathlib import Path

cwd = Path.cwd()
path_to_checkpoints = Path(cwd / "checkpoints")

# Specify the range of split files you want to process
start_range = 0
end_range = 3000

# Loop through the split files and execute the command for each one
for split_start in range(start_range, end_range, 1000):
    split_stop = split_start + 999
    checkpoint_file = path_to_checkpoints / f"last_successful_index_{split_start}.txt"

    # Check if the checkpoint file exists
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            last_successful_index = int(f.read())

        # Check if the last successful index is less than the stop value
        if last_successful_index < split_stop:
            # Add 2 to the last successful index and update the checkpoint file
            last_successful_index += 2
            with open(checkpoint_file, 'w') as f:
                f.write(str(last_successful_index))
        elif last_successful_index == split_stop:
            print(f"Skipping split_{split_start}-{split_stop}.npy as it's already completed.")
        else:
            print(f"Error: last_successful_index ({last_successful_index}) > split_stop ({split_stop}). Stopping the script.")
            break
    # Run the models_generator.py script regardless of the checkpoint file's existence
    command = f"nohup python models_generator.py split_{split_start}-{split_stop}.npy &"
    os.system(command)