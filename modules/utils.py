import math
import random
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from astropy.io import ascii


def plot_random_sample(path_to_models, n_plots_per_row=3):
    """
    Reads a random sample of ASCII files and plots them in an array of plots.

    Parameters:
    - path_to_models (Path): Path to the directory containing the ASCII files.
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
    ascii_files = list(path_to_models.glob("model_*.ipac"))

    # Randomly select n_plots_per_row^2 files from the list
    selected_files = random.sample(ascii_files, n_plots_per_row**2)

    # Calculate the number of rows needed
    n_rows = int(math.ceil(len(selected_files) / n_plots_per_row))

    # Create subplots for the selected number of plots
    fig, axes = plt.subplots(n_rows, n_plots_per_row, figsize=(15, 4 * n_rows))

    # Iterate through selected files and plot each one
    for i, file_path in enumerate(selected_files):
        # Read the table from the ASCII file
        table = ascii.read(file_path)

        # Extract energy and flux from the table
        energy = table['Energy']
        flux = table['Flux']

        # Get the name of the model from the file path
        model_name = file_path.stem

        # Calculate the row and column index for the subplot
        row_index = i // n_plots_per_row
        col_index = i % n_plots_per_row

        # Plot the data on the corresponding subplot
        axes[row_index, col_index].plot(energy, flux, label=f"Model: {model_name[:10]}")
        axes[row_index, col_index].set_xlabel('Energy (keV)')
        axes[row_index, col_index].set_ylabel('Flux (cm^2/s/keV)')
        axes[row_index, col_index].legend()

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

# Example usage:
# path_to_models = Path("/path/to/models")
# plot_random_sample(path_to_models, n_plots_per_row=3)

def process_ipac_files(models_folder_path):
    """
    Process .ipac files in the specified directory.

    This function reads data from .ipac files, extracts parameters from comments in the metadata,
    and creates a Pandas DataFrame for each model. It then concatenates all DataFrames into a
    single DataFrame, and groups the data by parameters while aggregating arrays.

    Parameters:
    - models_folder_path (str): Path to the directory containing the .ipac files.

    Returns:
    - grouped_data (pd.DataFrame): The grouped and aggregated data stored in a DataFrame.
    """
    # Validate that the provided path is a directory
    models_folder_path = Path(models_folder_path)
    if not models_folder_path.is_dir():
        raise ValueError("Invalid directory path. Please provide a valid directory path.")

    # Create a list to store individual DataFrames for each model
    data_frames = []

    # Iterate over each .ipac file in the specified directory
    for filepath in models_folder_path.glob("*.ipac"):
        # Read data from the .ipac file using astropy.ascii
        table = ascii.read(filepath, format='ipac')

        # Extract parameters from the comments in the metadata
        parameters = {}
        for comment in table.meta['comments']:
            if 'Parameter names' in comment:
                # Extract parameter names and values
                param_names = comment.split(': ')[-1].split(', ')
            elif 'Values' in comment:
                # Convert parameter values to float and store in a list
                param_values = [float(value) for value in comment.split(': ')[-1].split(', ')]

        # Create a dictionary mapping parameter names to their values
        parameters = dict(zip(param_names, param_values))

        # Extract energy and flux data from the table
        energy = table['Energy']
        flux = table['Flux']

        # Create a DataFrame for this model with Energy and Flux as lists
        model_data = pd.DataFrame({'Energy': [energy], 'Flux': [flux]})

        # Add parameters as columns in the DataFrame
        for param_name, param_value in parameters.items():
            model_data[param_name] = float(param_value)

        # Append the DataFrame for this model to the list
        data_frames.append(model_data)

    # Concatenate all DataFrames into a single DataFrame
    full_data = pd.concat(data_frames, ignore_index=True)

    # Group by parameters and aggregate arrays by summing them
    grouped_data = full_data.groupby(param_names).agg({'Energy': 'sum', 'Flux': 'sum'}).reset_index()

    return grouped_data

# Example usage:
# models_folder_path = "/path/to/models"
# result = process_ipac_files(models_folder_path)
# print(result)
