import h5py
import numpy as np
from pathlib import Path

from modules.utils import process_json_files_batch

def combine_hdf5_files(file_pattern):
    """
    Combine data from multiple HDF5 files.

    Parameters:
    - file_pattern (str): The Path for the HDF5 filenames.

    Returns:
    - combined_flux (np.ndarray): Combined flux data from all files.
    - combined_params (np.ndarray): Combined parameter data from all files.
    """
    combined_flux = []
    combined_params = []

    # Iterate over each .json file in the specified directory
    for i in np.sort(list(file_pattern.glob("*.h5"))):
        with h5py.File(i, 'r') as hf:
            combined_flux.extend(hf['flux'][:])
            combined_params.extend(hf['params'][:])

    return np.array(combined_flux, dtype=object), np.array(combined_params, dtype=object)

path_to_models = Path(Path.cwd() / 'models')
path_to_logs = Path(Path.cwd() / 'logs')
path_to_checkpoints = Path(Path.cwd() / 'checkpoints')

for i, (flux, params) in enumerate(process_json_files_batch(path_to_models)):
    with h5py.File(path_to_checkpoints / f'batch_{i}.h5', 'w') as hf:
        hf.create_dataset('flux', data=flux)
        hf.create_dataset('params', data=params)

# Example usage
combined_flux, combined_params = combine_hdf5_files(path_to_checkpoints)
