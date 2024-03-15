"""
Script Description:
This script performs spectral analysis by loading models and samples, generating synthetic
spectra, predicting fluxes using a deep learning model, and plotting the true versus predicted
fluxes for a series of spectra.

Dependencies:
- numpy, joblib, pathlib, xspec, tensorflow/keras, matplotlib, custom modules (logging_config, network, variables)

Steps:
1. Set up logging configuration.
2. Load spectral model and data samples.
3. Generate synthetic spectra based on the spectral model and parameters.
4. Load a deep learning model and predict fluxes for the generated spectra.
5. Plot the true versus predicted fluxes.
6. Save the last plot to a specified directory.

Output:
- A set of plots comparing true and predicted fluxes.
- Logs are saved to a specified log file for debugging and tracking.
"""
import numpy as np
from joblib import load
from xspec import FakeitSettings, AllModels, AllData, Model, Plot
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

from logging_config import logging_conf
from modules.network import r_squared
from modules.variables import (
    path_to_logs, 
    path_to_data, 
    path_to_plots,
    path_to_results
)

# Define data and smoothing points
data = 'models_100k'
NN = 'GRU'
arch = '256x4'
n_points = 0

# Set up paths for logs and models
path_to_logs.mkdir(parents=True, exist_ok=True)
path_to_plots.mkdir(parents=True, exist_ok=True)
result_dir = path_to_results / NN / arch
model_file_path = result_dir / f'{NN}_model_norm.h5'
path_to_data_scaler = path_to_data / data
path_to_files = path_to_data / 'files'

# Set up log configuration and create a logger for the fit
logger = logging_conf(path_to_logs, f"result_plot.log")
# Debug: Log the start of the script
logger.debug("Script started.")
logger.debug(f"File to check: {data}")
logger.debug(f"Model path: {model_file_path}")

fluxes = []

model_name = "TBabs*(rdblur*rfxconv*comptb + diskbb + comptb)"
fs1 = FakeitSettings(response=path_to_files.__str__()+"/ni5050300117mpu7.rmf", arf=path_to_files.__str__()+"/ni5050300117mpu7.arf", exposure="1e5", fileName='test.fak')

logger.debug("loading the sampled parameters")
sample_scaled = np.load(path_to_files / 'complete_sample.npy')
relevant_par =  np.load(path_to_files / 'relevant_par.npy')

# Load scalers and normalize parameters if needed
if 'norm' in model_file_path.name:
    logger.debug("loading the scalers to normalize parameters")
    # Load the scalers
    loaded_scalers = {}
    parameters = np.zeros_like(sample_scaled)
    for i in range(sample_scaled.shape[1]):
        loaded_scalers[i] = load(path_to_data_scaler / f'scaler_{i}.joblib')
    for i in range(sample_scaled.shape[1]):        
        parameters[:, i] = loaded_scalers[i].fit_transform(sample_scaled[:, i].reshape(-1,1)).flatten()

# Invert the log10 of these components
log_index = [12]
for i in range(sample_scaled.shape[0]):
    for j in log_index:
        sample_scaled[i, j] =  pow(10, sample_scaled[i, j])

# Iterate through scaled samples to set up and save models
for index, params in enumerate(sample_scaled):
    logger.debug(f"Creating spectrum # {index+1}")
    # Clear existing XSPEC models and data
    AllModels.clear()
    AllData.clear()

    # Initialize the model
    m = Model(model_name)

    # Changing default frozen parameters to unfrozen
    m.rdblur.Betor10.frozen = False
    m.rdblur.Rout_M.frozen = True
    m.rdblur.Rin_M.frozen = False
    m.rfxconv.Fe_abund.frozen = False
    m.comptb.gamma.frozen = True
    m.comptb.delta.frozen = True
    m.comptb.log_A.frozen = True

    m.rdblur.Rout_M.values = 1000
    m.comptb.delta.values = 0
    m.comptb.log_A.values = 8

    m.rfxconv.cosIncl.link = "COSD(5)"
    # Linking comptb_6 (refletion) parameters to comptb (comptb)
    start = 20  # Number of the first parameter of comptb_6
    for i in range(start, start + len(m.comptb_6.parameterNames)):
        m(i).link = str(i-9) # 9 is the separation between comptb and comptb_6

    # Add the model to the spectral analysis system and set parameters
    AllModels.setPars(m, {int(relevant_par[j]):params[j] for j in range(len(relevant_par))})
    AllData.fakeit(1, fs1)
    AllData.ignore('**-0.3')
    AllData.ignore('10.-**')
    
    # Set up the energy range of interest for plotting
    Plot.device = "/null"
    Plot.xAxis = "keV"
    Plot.show()
    Plot('data')
    energy = Plot.x()
    flux = Plot.model()
    fluxes.append(flux)
    
# Load the deep learning model
logger.debug("Loading the deep learning model")
model = load_model(model_file_path, custom_objects={'r_squared': r_squared})

# Load the saved scaler
flux_scaler = load(path_to_data_scaler / 'flux_scaler.joblib')
predicted = []

if 'norm' in model_file_path.name:
    looper = parameters
else:
    looper = sample_scaled
for par in looper:
    # Make predictions
    if 'norm' not in model_file_path.name:
        logger.debug('re-do the log10')
        log_index = [12]
        for i in log_index:
            par[i] = np.log10(par[i])
    prediction = model.predict(par.reshape(-1, par.shape[0]))
    flux_predicted = flux_scaler.inverse_transform(prediction)
    predicted.append(flux_predicted)

logger.debug('Printing the model summary:')
for layer in model.layers:
    logger.debug(f'{layer.name}')
    logger.debug('Layer Configuration:')
    logger.debug(f'{layer.get_config()}')
    logger.debug('Input Shape:')
    logger.debug(f'{layer.input_shape}')
    logger.debug('Output Shape:')
    logger.debug(f'{layer.output_shape}')
    if hasattr(layer, 'activation'):
        logger.debug('Activation Function:')
        logger.debug(f'{layer.activation.__name__}')

logger.debug(f'Optimizer: {model.optimizer}')
logger.debug(f'Loss: {model.loss}')
if hasattr(model, 'metrics'):
    logger.debug(f'Metrics: {[m.name for m in model.metrics]}')

# Create subplots for the selected number of plots
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
# Plotting
for idx, true_flux in enumerate(fluxes):
    
    # Calculate the row and column index for the subplot
    row_index = idx // 3
    col_index = idx % 3
    
    if n_points:
        energy_reshape = (np.array(energy).reshape(-1, n_points).mean(axis=1)).tolist()
        true_flux_reshape = (np.array(true_flux).reshape(-1, n_points).mean(axis=1)).tolist()
    else:
        energy_reshape = energy
        true_flux_reshape = true_flux
    # Plot the data on the corresponding subplot
    axes[row_index, col_index].plot(energy_reshape, true_flux_reshape, label="True")
    axes[row_index, col_index].plot(energy_reshape, predicted[idx][0], label='Predicted')
    axes[row_index, col_index].set_xlabel('Energy (keV)')
    axes[row_index, col_index].set_ylabel('Flux [counts / (keV s)]')
    axes[row_index, col_index].set_xscale('log')
    axes[row_index, col_index].set_yscale('log')
    axes[row_index, col_index].legend()    

# Save the last plot to the specified directory
plot_path = path_to_plots / f"{NN}_{arch}_results.png"
plt.savefig(plot_path)
logger.debug(f"Final plot saved to: {plot_path}")
logger.debug("Script ended.")  # Log the end of the script    