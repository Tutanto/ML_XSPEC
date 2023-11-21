"""
Script for Reading and Processing .ipac Files

This script reads data from .ipac files, extracts parameters from comments in the metadata,
and creates a Pandas DataFrame for each model. It then concatenates all DataFrames into a
single DataFrame, and groups the data by parameters while aggregating arrays.

Dependencies:
- astropy
- pathlib
- pandas

Usage:
1. Place .ipac files in the "models" directory.
2. Run the script.

Output:
- The grouped and aggregated data is stored in the 'grouped_data' DataFrame.

Author: Antonio Tutone
Date: 21/11/2023
"""

from astropy.io import ascii
from pathlib import Path
import pandas as pd

# Set the path to the directory containing the .ipac files
data_directory = Path(Path.cwd() / "models")

# Create a list to store individual DataFrames for each model
data_frames = []

# Iterate over each .ipac file in the specified directory
for filepath in data_directory.glob("*.ipac"):
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