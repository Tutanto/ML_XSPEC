import sys
import h5py
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import custom modules
from modules.utils import (
    process_pickle_files_batch,
    combine_hdf5_files)

from modules.variables import path_to_all_models, path_to_batches

def plot(input_path_to_models):
    path_to_models = path_to_all_models / input_path_to_models

    # Ensure the path to batches exists
    path_to_batches.mkdir(parents=True, exist_ok=True)

    # Read the pickle models in batches
    for i, (flux, params) in enumerate(process_pickle_files_batch(path_to_models)):
        with h5py.File(path_to_batches / f'batch_{i}.h5', 'w') as hf:
            hf.create_dataset('flux', data=flux)
            hf.create_dataset('params', data=params)

    # Put all the batches together
    all_flux_values, all_parameters = combine_hdf5_files(path_to_batches)
    # Convert y_train to a float data type
    all_parameters = np.array(all_parameters, dtype=float)

    # Read one model
    model = list(path_to_models.glob("*pkl"))[0]
    with open(model, 'rb') as file:
        data = pickle.load(file)
        par_names = list(data['parameters'].keys())

    # Setting up the plot
    plt.figure(figsize=(20, 10))
    for i, name in enumerate(par_names):
        plt.subplot(3, 6, i+1)
        sns.histplot(all_parameters[:, i], kde=True)
        plt.title(name)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: script.py <path_to_models>")
        sys.exit(1)

    input_path_to_models = sys.argv[1]
    plot(input_path_to_models)
