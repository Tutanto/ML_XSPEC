import h5py
import pickle
import random
import numpy as np

from pathlib import Path
    
def remove_uniform_columns(data):
    """
    Removes columns with uniform values across all rows from a numpy array.

    This function identifies and removes columns where all elements are identical,
    as such columns typically do not provide useful information for analysis.

    Parameters:
    data (numpy.ndarray): The array from which to remove uniform columns.

    Returns:
    numpy.ndarray: The array with uniform columns removed.
    list: List of indices of the columns that were removed.
    """
    columns_to_remove = []
    for i in range(data.shape[1]):
        if np.all(data[:, i] == data[0, i]):
            columns_to_remove.append(i)
    return np.delete(data, columns_to_remove, axis=1), columns_to_remove

def process_pickle_files_batch(models_folder_path, batch_size=1000):
    """
    Process .pkl files in batches.

    Parameters:
    - models_folder_path (Path): Path object pointing to the directory containing the .pkl files.
    - batch_size (int): Number of files to process in each batch.

    Yields processed data in batches.
    """
    if not isinstance(models_folder_path, Path) or not models_folder_path.is_dir():
        raise ValueError("Invalid directory path. Please provide a valid Path object pointing to a directory.")

    batch_flux = []
    batch_params = []
    count = 0

    # Iterate over each .pkl file in the specified directory
    sorted_models = list(models_folder_path.glob("*.pkl"))
    random.shuffle(sorted_models)
    for filepath in sorted_models:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)

        batch_flux.append(data['flux (counts / keV s)'])
        batch_params.append(list(data['parameters'].values()))

        count += 1
        if count >= batch_size:
            yield (batch_flux, batch_params)
            batch_flux, batch_params = [], []
            count = 0

    # Yield any remaining data
    if batch_flux and batch_params:
        yield (batch_flux, batch_params)
    
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