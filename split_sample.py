import datetime
import numpy as np
from pathlib import Path
from modules.logging_config import logging_conf

def load_and_split_npy(file_path, file_name, num_splits):
    """
    Loads an ndarray from a .npy file, splits it into equal parts, and saves each part as a new .npy file.

    Parameters:
    file_path (str): The path to the .npy file.
    num_splits (int): The number of equal parts to split the array into.
    """
    # Load the ndarray
    array = np.load(file_path / file_name)

    # Calculate the number of rows in each split
    split_size = len(array) // num_splits

    # Split the array and save each part
    for i in range(num_splits):
        start_index = i * split_size
        end_index = start_index + split_size if i < num_splits - 1 else len(array)
        split_array = array[start_index:end_index]

        # Save each split array to a file
        split_file_name = f"split_{start_index}-{end_index - 1}.npy"
        np.save(file_path / split_file_name, split_array)
        logger.debug(f"Saved: {split_file_name}")

# Usage example
if __name__ == "__main__":
    sample_scaled_file_name = 'complete_sample.npy'  # Replace with your file path
    num_splits = 10

    # Set up paths for logs and models
    cwd = Path.cwd()
    path_to_logs = Path(cwd / "logs")
    path_to_logs.mkdir(parents=True, exist_ok=True)
    path_to_samples = Path(cwd / "samples")
    path_to_samples.mkdir(parents=True, exist_ok=True)

    # Get the current date and time
    t_start = datetime.datetime.now()
    # Format the current date and time as a string
    timestamp = t_start.strftime("%Y-%m-%d_%H-%M-%S")

    # Set up log configuration and create a logger for the fit
    logger = logging_conf(path_to_logs, f"sample_splitter_{timestamp}.log")
    # Debug: Log the start of the script
    logger.debug("Script started.")
    logger.debug(f"File to split: {sample_scaled_file_name}")
    logger.debug(f"Number of splits: {num_splits}")

    load_and_split_npy(path_to_samples, sample_scaled_file_name, num_splits)

    # Get the current date and time
    t_stop = datetime.datetime.now()
    logger.debug(f"execution time: {t_stop - t_start}")
    # Debug: Log the end of the script
    logger.debug("Script completed.")
