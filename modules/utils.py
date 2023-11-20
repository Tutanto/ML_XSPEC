import matplotlib.pyplot as plt
import random
from pathlib import Path
from astropy.io import ascii
import math

def plot_random_sample(path_to_models, n_plots_per_row=3):
    """
    Reads a random sample of ASCII files and plots them in an array of plots.

    Parameters:
    - path_to_models (Path): Path to the directory containing the ASCII files.
    - n_plots_per_row (int): Number of plots to display per row.

    Returns:
    - None
    """

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
