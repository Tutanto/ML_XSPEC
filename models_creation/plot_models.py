import sys

# Import custom modules
from modules.utils import plot_random_sample
from modules.variables import path_to_all_models

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
    
    if len(sys.argv) < 3:
        print("Usage: script.py <n_plots_per_row> <path_to_models>")
        sys.exit(1)
        
    # Get the input value from command line arguments
    n_plots_per_row_input = sys.argv[1]
    input_path_to_models = sys.argv[2]


    # Set up paths for logs and models
    path_to_models = path_to_all_models / input_path_to_models

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
