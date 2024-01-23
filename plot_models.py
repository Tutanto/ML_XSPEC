import sys
from pathlib import Path

# Import custom modules
from modules.utils import plot_random_sample

def is_valid_number(n, max_value):
    """
    Check if the given input 'n' is a valid integer and less than 'max_value'.

    Parameters:
    n (str): The input string to validate.
    max_value (int): The upper limit for the valid integer range.

    Returns:
    bool: True if 'n' is a valid integer and less than 'max_value', False otherwise.
    """
    try:
        n = int(n)
        if n < max_value:
            return True
        else:
            print(f"Error: The number of plots per row must be less than {max_value}")
            return False
    except ValueError:
        print("Error: Please enter a valid integer for the number of plots per row.")
        return False

if __name__ == "__main__":
    
    # Get the input value for n_plots_per_row from command line arguments
    n_plots_per_row_input = sys.argv[1]

    # Set up paths for logs and models
    cwd = Path.cwd()
    path_to_models = Path(cwd / 'models')

    # Get the count of files in models directory to set an upper limit for n_plots_per_row
    num_files = len(list(path_to_models.glob('*')))

    # Validate the n_plots_per_row input
    if is_valid_number(n_plots_per_row_input, num_files):
        n_plots_per_row = int(n_plots_per_row_input)
        # Plot random sample of generated models
        plot_random_sample(path_to_models, n_plots_per_row=n_plots_per_row)
    else:
        # Exit the program if the input is not valid
        sys.exit(1)
