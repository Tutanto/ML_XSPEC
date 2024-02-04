import h5py
import datetime
import numpy as np
from joblib import dump
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

from modules.utils import (
    process_json_files_batch, 
    combine_hdf5_files, 
    remove_uniform_columns
)
from modules.logging_config import logging_conf

cwd = Path.cwd()
path_to_models = cwd / 'all_models' / 'models_0.5-20_10k'
path_to_logs = cwd / 'logs'
path_to_batches = cwd / 'batches'
path_to_batches.mkdir(parents=True, exist_ok=True)
path_to_data = cwd / 'data'
path_to_data.mkdir(parents=True, exist_ok=True)

# Get the current date and time
t_start = datetime.datetime.now()
# Format the current date and time as a string
timestamp = t_start.strftime("%Y-%m-%d_%H-%M-%S")
# Set up log configuration and create a logger for the fit
logger = logging_conf(path_to_logs, f"preprocessing_{timestamp}.log")
# Debug: Log the start of the script
logger.debug("Script started.")
logger.debug(f"Models: {path_to_models}")

# Check if any 'batch_*.h5' files already exist
logger.debug("Reading the jsons models file in batches...")
existing_batches = list(path_to_batches.glob('batch_*.h5'))
if not existing_batches:
    # Read the json models in batches
    for i, (flux, params) in enumerate(process_json_files_batch(path_to_models)):
        with h5py.File(path_to_batches / f'batch_{i}.h5', 'w') as hf:
            hf.create_dataset('flux', data=flux)
            hf.create_dataset('params', data=params)
logger.debug(f"Batches created")

# Put all the batches together
all_flux_values, all_parameters = combine_hdf5_files(path_to_batches)
logger.debug("Data loaded")

# Convert parameters to a float data type
all_parameters = np.array(all_parameters, dtype=float)
# Function to remove columns with identical values across all rows
relevant_parameters, removed_columns = remove_uniform_columns(all_parameters)
logger.debug("Removed columns with identical values across all rows")

logger.debug("Modified Data:")
logger.debug(f"{relevant_parameters}")
logger.debug("Removed Columns Indices:")
logger.debug(f"{removed_columns}")

# Normalize flux values
flux_scaler = MinMaxScaler(feature_range=(0, 1))
Out_norm = flux_scaler.fit_transform(all_flux_values.reshape(-1, all_flux_values.shape[-1])).reshape(all_flux_values.shape)
logger.debug("Flux normalized")

# Find the indices of rows that contain NaN
indices_with_nan = np.any(np.isnan(Out_norm), axis=1)
# Count the rows to be removed
rows_to_remove = np.sum(indices_with_nan)
# Remove the rows that contain NaN
Inp = relevant_parameters[~indices_with_nan]
Out_norm = Out_norm[~indices_with_nan]
logger.debug(f"Number of rows that contain NaN removed: {rows_to_remove}")

# Saving the normalized datasets and scalers to disk
np.save(path_to_data / 'Inp.npy', Inp)
np.save(path_to_data / 'Out_norm.npy', Out_norm)
# Save the scaler
dump(flux_scaler, path_to_data / 'flux_scaler.joblib')
logger.debug(f"Data saved in: {path_to_data}")
logger.debug("Script completed.")
