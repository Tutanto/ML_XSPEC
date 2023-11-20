from astropy.io import ascii
from pathlib import Path
import pandas as pd

data_directory = Path(Path.cwd() / "modelspa")
data_frames = []

for filepath in data_directory.glob("*.ipac"):
    # Read data from the .ipac file
    table = ascii.read(filepath, format='ipac')

    # Extract parameters from the comments
    parameters = {}
    for comment in table.meta['comments']:
        if 'Parameter names' in comment:
            # Extract parameter names and values
            param_names = comment.split(': ')[-1].split(', ')
            param_values = table.meta['comments'][comment].split(', ')
            parameters = dict(zip(param_names, param_values))

    # Get energy and flux data
    energy = table['Energy']
    flux = table['Flux']

    # Create a DataFrame for this model
    model_data = pd.DataFrame({'Energy': energy, 'Flux': flux})

    # Add parameters as columns in the DataFrame
    for param_name, param_value in parameters.items():
        model_data[param_name] = float(param_value)

    data_frames.append(model_data)

# Concatenate all DataFrames into a single DataFrame
full_data = pd.concat(data_frames, ignore_index=True)