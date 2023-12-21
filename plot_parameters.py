import h5py
import json
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from modules.utils import (
    process_json_files_batch,
    combine_hdf5_files)

path_to_models = Path(Path.cwd() / 'models')
path_to_batches = Path(Path.cwd() / 'batches')
path_to_batches.mkdir(parents=True, exist_ok=True)

# Read the json models in batches
for i, (flux, params) in enumerate(process_json_files_batch(path_to_models)):
    with h5py.File(path_to_batches / f'batch_{i}.h5', 'w') as hf:
        hf.create_dataset('flux', data=flux)
        hf.create_dataset('params', data=params)

# Put all the batches together
all_flux_values, all_parameters = combine_hdf5_files(path_to_batches)
# Convert y_train to a float data type
all_parameters = np.array(all_parameters, dtype=float)

# Read one model
model = list(path_to_models.glob("*json"))[0]
with open(model, 'r') as file:
    data = json.load(file)
    par_names = list(data['parameters'].keys())

# Setting up the plot
plt.figure(figsize=(20, 10))
for i, name in enumerate(par_names):
    plt.subplot(3, 6, i+1)
    sns.histplot(all_parameters[:, i], kde=True)
    plt.title(name)

plt.tight_layout()
plt.show()