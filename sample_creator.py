'''
Script: Spectral Model Sample Creator

Description:
    This script generates and saves parameter samples for spectral models using the XSPEC astrophysical data analysis package. 
    It uses a Latin Hypercube sampling technique to create samples of model parameters, scales these samples according to specified parameter bounds, 
    and saves the scaled samples and relevant parameters for further use. It also configures logging to record the script's operations.

Dependencies:
    - Python 3.x
    - XSPEC (for model creation and parameter handling)
    - NumPy (for array operations and saving samples)
    - SciPy (for Latin Hypercube sampling and scaling)
    - pathlib (for file and directory operations)
    - Custom modules: utils, logging_config

Outputs:
    - Scaled parameter samples saved in a .npy file.
    - Relevant parameter indices saved in a .npy file.
    - Logs of the script's operation.

Author: Antonio Tutone
Date: 14/12/2023

Usage:
    Run the script normally. No command-line arguments are required.
    Ensure all dependencies are installed and the XSPEC environment is properly configured.
'''

# Import necessary libraries
import datetime
import numpy as np
from xspec import Model
from pathlib import Path
from scipy.stats import qmc

# Import custom modules
from modules.utils import (
    generate_latin_hypercube, 
    indices_satisfying_condition
    )
from modules.logging_config import logging_conf

# Set the size of the Dataset
N = 10000
# Set checkpoint file names
sample_scaled_file_name = 'complete_sample.npy'

# Set up paths for logs and models
cwd = Path.cwd()
path_to_logs = Path(cwd / "logs")
path_to_logs.mkdir(parents=True, exist_ok=True)
path_to_samples = Path(cwd / "samples")
path_to_samples.mkdir(parents=True, exist_ok=True)
sample_scaled_file_path = path_to_samples / sample_scaled_file_name

# Get the current date and time
t_start = datetime.datetime.now()
# Format the current date and time as a string
timestamp = t_start.strftime("%Y-%m-%d_%H-%M-%S")

# Set up log configuration and create a logger for the fit
logger = logging_conf(path_to_logs, f"sample_creator_{timestamp}.log")

# Debug: Log the start of the script
logger.debug("Script started.")

# Create the model
model_name = "TBabs*(rdblur*rfxconv*comptb + diskbb + comptb)"
model = Model(model_name)
logger.debug(f"Model name: {model_name}")
logger.debug(f"Model used: {model.componentNames}")

# Changing default frozen parameters to unfrozen
model.rdblur.Betor10.frozen = False
model.rdblur.Rout_M.frozen = True
model.rdblur.Rin_M.frozen = False
model.rfxconv.Fe_abund.frozen = False
model.comptb.gamma.frozen = True
model.comptb.delta.frozen = True
model.comptb.log_A.frozen = True

# Fixing values, upper and lower limits
model.TBabs.nH.values = [1.0, 0.01, 0.01, 0.01, 10.0, 10.0]
model.rdblur.Betor10.values = [-2, 0.02, -10.0,-10.0, 0,0]
model.rdblur.Rin_M.values = [10.0, -0.1, 6.0, 6.0, 150.0, 10000.0]
model.rfxconv.rel_refl.values = [-1.0, 0.01, -1, -1, 0, 0]
model.rfxconv.log_xi.values = [1.0, 0.01, 1.0, 1.0, 4.0, 4.0]
model.comptb.alpha.values = [2, 0.02, 0, 0.1, 3, 3]
model.comptb.kTe.values = [5, 0.05, 0.2, 2, 1000, 1000]
model.comptb.kTs.values = [1.0, 0.01, 0.1, 0.15, 2, 10.0]
model.comptb.norm.values = [1.0, 0.01, 0.1, 0.1, 1.e0, 1e+24]
model.diskbb.Tin.values = [1.0, 0.01, 0.1, 0.1, 2, 1000.0]
model.diskbb.norm.values = [1.0, 0.01, 0.1, 0.1, 1.e4, 1e+24]

# Linking the parameters
model.rfxconv.cosIncl.link = "COSD(5)"
# Linking comptb_6 (refletion) parameters to comptb (comptb)
start = 20  # Number of the first parameter of comptb_6
for i in range(start, start + len(model.comptb_6.parameterNames)):
    model(i).link = model(i-9) # 9 is the separation between comptb and comptb_6

logger.debug(model.show())

# Collect the relevant parameter (the ones not frozen or linked)
relevant_par = []
for n_par in range(1, model.nParameters + 1):
    if not model(n_par).frozen and not model(n_par).link:
        relevant_par.append(n_par)

# Extract lower and upper bounds, and parameter names for scaling
l_bounds, u_bounds, par_names = [], [], []

# Compute the log10 of these components
log_index = [1, 15, 19]
for n_par in relevant_par:
    name = model(n_par).name
    # Append the values
    par_names.append(name)
    if n_par in log_index:
        l_bounds.append(np.log10(model(n_par).values[3])) #bot
        u_bounds.append(np.log10(model(n_par).values[4])) #top
    else:
        l_bounds.append(model(n_par).values[3]) #bot
        u_bounds.append(model(n_par).values[4]) #top

# Define a condition function (e.g., elements greater than 5)
condition_func = lambda x: x == 'kTs' or x == 'kTe' or x == 'Tin'

# Get the indices where the condition is satisfied
result_indices = indices_satisfying_condition(par_names, condition_func)

# Create a Latin Hypercube sampler for model parameters
sample = generate_latin_hypercube(d=len(relevant_par), linked=result_indices, n=N)
logger.debug("Generated Latin Hypercube samples")
logger.debug(f"Components in the model: {par_names}")

# Scale the sample to fit parameter bounds
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
# Apply the condition kTe < 6 then alpha < 1.5
for i in range(sample_scaled.shape[0]):
    if pow(10, sample_scaled[i][9]) < 6 and sample_scaled[i][8] > 1.5:
        sample_scaled[i][8] = np.random.uniform(0.1, 1.5)
    elif pow(10, sample_scaled[i][9]) > 6 and sample_scaled[i][8] < 1.5:
        sample_scaled[i][8] = np.random.uniform(1.5, 3)
    
logger.debug("Scaled samples to parameter bounds")

np.save(sample_scaled_file_path, sample_scaled)
np.save(path_to_samples / "relevant_par.npy", relevant_par)

# Get the current date and time
t_stop = datetime.datetime.now()
logger.debug(f"execution time: {t_stop - t_start}")
# Debug: Log the end of the script
logger.debug("Script completed.")
