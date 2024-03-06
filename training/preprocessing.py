import h5py
import datetime
import numpy as np
from joblib import dump

from sklearn.preprocessing import MinMaxScaler

from modules.utils import (
    process_json_files_batch, 
    combine_hdf5_files, 
    remove_uniform_columns
)
from modules.variables import (
    path_to_logs,
    path_to_data,
    path_to_batches,
    path_to_all_models
)

from logging_config import logging_conf

# Define the current working directory and paths for models, logs, batches, and data
path_to_models = path_to_all_models / 'models_0.5-20_100k'
path_to_logs.mkdir(parents=True, exist_ok=True)
path_to_batches.mkdir(parents=True, exist_ok=True)  # Create the batches directory if it doesn't exist
path_to_data.mkdir(parents=True, exist_ok=True)     # Create the data directory if it doesn't exist

# Initialize the logging process with timestamp
t_start = datetime.datetime.now()
timestamp = t_start.strftime("%Y-%m-%d_%H-%M-%S")
logger = logging_conf(path_to_logs, f"preprocessing_{timestamp}.log")
logger.debug("Script started.")
logger.debug(f"Models: {path_to_models}")

# Check for existing batch files and process new batches if necessary
logger.debug("Reading the json models file in batches...")
existing_batches = list(path_to_batches.glob('batch_*.h5'))
if not existing_batches:
    for i, (flux, params) in enumerate(process_json_files_batch(path_to_models)):
        with h5py.File(path_to_batches / f'batch_{i}.h5', 'w') as hf:
            hf.create_dataset('flux', data=flux)
            hf.create_dataset('params', data=params)
logger.debug(f"Batches created")

# Combine all batch files into a single dataset
all_flux_values, all_parameters = combine_hdf5_files(path_to_batches)
logger.debug("Data loaded")

# Convert parameters to float and remove uniform columns
all_parameters = np.array(all_parameters, dtype=float)
relevant_parameters, removed_columns = remove_uniform_columns(all_parameters)
logger.debug("Removed columns with identical values across all rows")

logger.debug("Modified Data:")
logger.debug(f"{relevant_parameters}")
logger.debug("Removed Columns Indices:")
logger.debug(f"{removed_columns}")

# Normalize the parameters and flux values
scalers = {}
Inp_tmp = np.zeros_like(relevant_parameters)

# Iterate through columns (parameters) and apply normalization
for i in range(relevant_parameters.shape[1]):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    Inp_tmp[:, i] = scaler.fit_transform(relevant_parameters[:, i].reshape(-1, 1)).flatten()
    scalers[i] = scaler  # Store the scaler for future use
logger.debug("Parameters normalized")

# Normalize the flux values
flux_scaler = MinMaxScaler(feature_range=(0, 1))
Out_norm = flux_scaler.fit_transform(all_flux_values.reshape(-1, all_flux_values.shape[-1])).reshape(all_flux_values.shape)
logger.debug("Flux normalized")

# Identify and remove rows with NaN values
indices_with_nan = np.any(np.isnan(Out_norm), axis=1)
rows_to_remove = np.sum(indices_with_nan)
Inp = relevant_parameters[~indices_with_nan]
Out = all_flux_values[~indices_with_nan]
Inp_norm = Inp_tmp[~indices_with_nan]
Out_norm = Out_norm[~indices_with_nan]
logger.debug(f"Number of rows that contain NaN removed: {rows_to_remove}")

# Save the normalized datasets and scalers to disk
np.save(path_to_data / 'Inp.npy', Inp)
np.save(path_to_data / 'Inp_norm.npy', Inp_norm)
np.save(path_to_data / 'Out.npy', Out)
np.save(path_to_data / 'Out_norm.npy', Out_norm)
for i, scaler in scalers.items():
    dump(scaler, path_to_data / f'scaler_{i}.joblib')
dump(flux_scaler, path_to_data / 'flux_scaler.joblib')
logger.debug(f"Data saved in: {path_to_data}")
logger.debug("Script completed.")
