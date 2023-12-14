import re
import math
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import qmc

def extract_number(filename):
    match = re.search(r'split_(\d+)-\d+\.npy', filename)
    if match:
        return int(match.group(1))
    else:
        return None

# Function to save the last successful index to a file
def save_last_successful_index(index, file_path):
    with open(file_path, 'w') as file:
        file.write(str(index))

# Function to read the last successful index from a file
def read_last_successful_index(file_path):
    try:
        with open(file_path, 'r') as file:
            return int(file.read())
    except FileNotFoundError:
        return 0  # If the file does not exist, start from the beginning

def generate_latin_hypercube(d, linked, n=10, seed=None):
    """
    Generate a Latin Hypercube sample for model parameters.

    Parameters:
    - d (int): The dimensionality of the Latin Hypercube.
    - linked (list): A list representing linked parameters where the second parameter
                     should always be less than the first one.
    - n (int): The number of samples to generate. Default is 10.
    - seed (int): Seed for reproducibility. Default is None.

    Returns:
    - parameters (numpy.ndarray): The Latin Hypercube sample with shape (n, d).
    """
    # Input validation
    if not isinstance(d, int) or d <= 0:
        raise ValueError("The dimensionality 'd' must be a positive integer.")
    
    if not isinstance(linked, list) or len(linked) != 2:
        raise ValueError("The 'linked' parameter must be a list of two indices.")
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError("The number of samples 'n' must be a positive integer.")

    # Set the seed for reproducibility
    np.random.seed(seed)

    # Create a Latin Hypercube sampler for model parameters
    sampler = qmc.LatinHypercube(d=d)

    # Generate a Latin Hypercube sample
    parameters = sampler.random(n=n)

    # Ensure the second parameter is always less than the first one
    parameters[:, linked[1]] = parameters[:, linked[0]] * parameters[:, linked[1]]

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
    
    # Get a list of all ASCII files in the specified directory
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

def process_json_files(models_folder_path):
    """
    Process .json files in the specified directory.

    This function reads data from .json files, extracts parameters and arrays,
    and creates a Pandas DataFrame for each model. It then concatenates all
    DataFrames into a single DataFrame, and groups the data by parameters while
    aggregating arrays.

    Parameters:
    - models_folder_path (str): Path to the directory containing the .json files.

    Returns:
    - grouped_data (pd.DataFrame): The grouped and aggregated data stored in a DataFrame.
    """
    # Validate that the provided path is a directory
    models_folder_path = Path(models_folder_path)
    if not models_folder_path.is_dir():
        raise ValueError("Invalid directory path. Please provide a valid directory path.")

    # Create a list to store individual DataFrames for each model
    data_frames = []

    # Iterate over each .json file in the specified directory
    for filepath in models_folder_path.glob("*.json"):
        # Read data from the .json file
        with open(filepath, 'r') as file:
            data = json.load(file)

        # Extract parameters and arrays from the JSON data
        parameters = data['parameters']
        energy = data['energy (keV)']
        flux = data['flux (1 / keV cm^-2 s)']

        # Create a DataFrame for this model with Energy and Flux as lists
        model_data = pd.DataFrame({'Energy': energy, 'Flux': flux})

        # Add parameters as columns in the DataFrame
        for param_name, param_value in parameters.items():
            model_data[param_name] = float(param_value)

        # Append the DataFrame for this model to the list
        data_frames.append(model_data)

    # Concatenate all DataFrames into a single DataFrame
    full_data = pd.concat(data_frames, ignore_index=True)

    # Convert parameters.keys() to a list before using it in groupby
    # Group by parameters and aggregate arrays by collecting them in lists
    grouped_data = full_data.groupby(list(parameters.keys())).agg({
        'Energy': lambda x: list(x),
        'Flux': lambda x: list(x)
    }).reset_index()


    return grouped_data

# Example usage:
# models_folder_path = "/path/to/models"
# result = process_json_files(models_folder_path)
# print(result)