import re
import math
import json
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import qmc

def extract_number(filename):
    """
    Extracts a number from a filename based on a specific pattern.

    The function searches for a pattern 'split_X-Y.npy' in the filename,
    where X and Y are digits, and returns the value X as an integer.
    If the pattern is not found, it returns None.

    Parameters:
    filename (str): The filename from which to extract the number.

    Returns:
    int or None: The extracted number or None if the pattern is not found.
    """
    match = re.search(r'split_(\d+)-\d+\.npy', filename)
    if match:
        return int(match.group(1))
    else:
        return None
    
def save_last_successful_index(index, file_path):
    """
    Saves the last successful index to a file.

    This function writes the index to a specified file. It's used to track
    the progress of a process, allowing for resumption from the last saved point.

    Parameters:
    index (int): The index to be saved.
    file_path (str): Path to the file where the index will be saved.
    """
    with open(file_path, 'w') as file:
        file.write(str(index))
        
def read_last_successful_index(file_path):
    """
    Reads the last successful index from a file.

    Attempts to read an index from a specified file. If the file does not exist,
    it returns 0, indicating the start of a process.

    Parameters:
    file_path (str): Path to the file from which to read the index.

    Returns:
    int: The last successful index, or 0 if the file does not exist.
    """
    try:
        with open(file_path, 'r') as file:
            return int(file.read())
    except FileNotFoundError:
        return 0  # If the file does not exist, start from the beginning
    
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

def adjust_values(kTe, kTs, Tin):
    """
    Adjusts the values of kTe, kTs, and Tin based on certain conditions.

    This function modifies the values of kTs and Tin to ensure they are
    not greater than certain thresholds based on the value of kTe. The
    adjustments involve reducing kTs and Tin if they exceed these thresholds,
    ensuring that they are just below kTe's threshold.

    Parameters:
    kTe (float): Initial value of kTe.
    kTs (float): Initial value of kTs.
    Tin (float): Initial value of Tin.

    Returns:
    float, float, float: Adjusted values of kTe, kTs, and Tin.
    """
    val = 0.01
    if kTs >= pow(10, kTe):
        if (pow(10, kTe) - val) > 0:
            kTs = pow(10, kTe) - val
        else:
            kTs = pow(10, kTe)
    if Tin >= kTs:
        if (kTs - val) > 0:
            Tin = kTs - val
        else:
            Tin = kTs
    return kTe, kTs, Tin

def generate_latin_hypercube(d, linked, n=10, seed=None):
    """
    Generate a Latin Hypercube sample for model parameters.

    Parameters:
    - d (int): The dimensionality of the Latin Hypercube.
    - linked (list): A list representing linked parameters where the 3rd parameter
                     should always be less than the 1st one, and the 1st < 2nd.
    - n (int): The number of samples to generate. Default is 10.
    - seed (int): Seed for reproducibility. Default is None.

    Returns:
    - parameters (numpy.ndarray): The Latin Hypercube sample with shape (n, d).
    """
    # Input validation
    if not isinstance(d, int) or d <= 0:
        raise ValueError("The dimensionality 'd' must be a positive integer.")
    
    if not isinstance(linked, list) or len(linked) != 3:
        raise ValueError("The 'linked' parameter must be a list of 3 indices.")
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError("The number of samples 'n' must be a positive integer.")

    # Set the seed for reproducibility
    np.random.seed(seed)

    # Create a Latin Hypercube sampler for model parameters
    sampler = qmc.LatinHypercube(d=d)

    # Generate a Latin Hypercube sample
    parameters = sampler.random(n=n)

    # Ensure the second parameter is always less than the first one
    # If kTs is not less than kTe, set kTs to a value just below kTe
    # Indexes of kTe, kTs, and Tin in each sub-array
    idx_kTe = linked[1]  # Example index for kTe
    idx_kTs = linked[0]  # Example index for kTs
    idx_Tin = linked[2]  # Example index for Tin

    # Apply adjustments to each sub-array
    for sub_array in parameters:
        kTe, kTs, Tin = sub_array[idx_kTe], sub_array[idx_kTs], sub_array[idx_Tin]
        kTe, kTs, Tin = adjust_values(kTe, kTs, Tin)
        sub_array[idx_kTe], sub_array[idx_kTs], sub_array[idx_Tin] = kTe, kTs, Tin

    return parameters
# Example usage:
#latin_hypercube_sample = generate_latin_hypercube(n=10, seed=42)
#print("Latin Hypercube Sample:")
#print(latin_hypercube_sample)

def indices_satisfying_condition(my_list, condition_func):
    """
    Return indices of elements in the list satisfying a given condition.

    Parameters:
    - my_list (list): The input list.
    - condition_func (function): The condition function that takes an element as input
                                and returns a boolean indicating whether the condition is satisfied.

    Returns:
    - indices (list): The list of indices where the condition is satisfied.
    """
    # Input validation
    if not callable(condition_func):
        raise ValueError("The 'condition_func' parameter must be a callable function.")
    
    return [index for index, element in enumerate(my_list) if condition_func(element)]

def plot_random_sample(path_to_models, n_plots_per_row=3):
    """
    Reads a random sample of JSON files and plots them in an array of plots.

    Parameters:
    - path_to_models (Path): Path to the directory containing the JSON files.
    - n_plots_per_row (int): Number of plots to display per row.

    Returns:
    - None
    """
    
     # Validate that path_to_models is a valid directory
    if not isinstance(path_to_models, Path) or not path_to_models.is_dir():
        raise ValueError("Invalid directory path. Please provide a valid directory path.")

    # Validate that n_plots_per_row is a positive integer
    if not isinstance(n_plots_per_row, int) or n_plots_per_row <= 0:
        raise ValueError("Invalid value for n_plots_per_row. Please provide a positive integer.")
    
    # Get a list of all JSONS files in the specified directory
    json_files = list(path_to_models.glob("model_*.json"))

    # Randomly select n_plots_per_row^2 files from the list
    selected_files = random.sample(json_files, n_plots_per_row**2)

    # Calculate the number of rows needed
    n_rows = int(math.ceil(len(selected_files) / n_plots_per_row))

    # Create subplots for the selected number of plots
    fig, axes = plt.subplots(n_rows, n_plots_per_row, figsize=(15, 4 * n_rows))

    # Iterate through selected files and plot each one
    for i, file_path in enumerate(selected_files):
        # Read the JSON data from the file
        with open(file_path, 'r') as json_file:
            json_data = json.load(json_file)

        # Extract energy and flux from the JSON data (replace 'Energy' and 'Flux' with your actual keys)
        energy = json_data['energy (keV)']
        flux = json_data['flux (1 / keV cm^-2 s)']

        # Get the name of the model from the file path
        model_name = file_path.stem

        # Calculate the row and column index for the subplot
        row_index = i // n_plots_per_row
        col_index = i % n_plots_per_row

        # Plot the data on the corresponding subplot
        axes[row_index, col_index].plot(energy, flux, label=f"Model: {model_name[:10]}")
        axes[row_index, col_index].set_xlabel('Energy (keV)')
        axes[row_index, col_index].set_xscale('log')
        axes[row_index, col_index].set_ylabel('Flux (cm^2/s/keV)')
        axes[row_index, col_index].legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

# Example usage:
# path_to_models = Path("/path/to/models")
# plot_random_sample(path_to_models, n_plots_per_row=3)

def process_json_files_batch(models_folder_path, batch_size=1000):
    """
    Process .json files in batches.

    Parameters:
    - models_folder_path (Path): Path object pointing to the directory containing the .json files.
    - batch_size (int): Number of files to process in each batch.

    Yields processed data in batches.
    """
    if not isinstance(models_folder_path, Path) or not models_folder_path.is_dir():
        raise ValueError("Invalid directory path. Please provide a valid Path object pointing to a directory.")

    batch_flux = []
    batch_params = []
    count = 0

    # Iterate over each .json file in the specified directory
    sorted_models = list(models_folder_path.glob("*.json"))
    random.shuffle(sorted_models)
    for filepath in sorted_models:
        with open(filepath, 'r') as file:
            data = json.load(file)

        batch_flux.append(data['flux (1 / keV cm^-2 s)'])
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