"""
Script: model_generation_script.py

Description:
    This script generates and saves synthetic spectral models using the XSPEC library. It creates a Latin Hypercube
    sampler to sample parameters for a diskline model and then scales the samples to fit the parameter bounds.
    The script iterates through the scaled samples, sets up XSPEC models with the sampled parameters, and saves
    the generated models as IPAC format tables. Finally, it processes these IPAC files to create a grouped and
    aggregated DataFrame.

Dependencies:
    - Python 3.x
    - XSPEC (XSPEC models and data fitting library)
    - Astropy (for handling units and table creation)
    - SciPy (for Latin Hypercube sampling)
    - Matplotlib (for plotting)
    - Pandas (for data manipulation)

Usage:
    1. Ensure that all dependencies are installed.
    2. Run the script.

Outputs:
    - IPAC format tables containing synthetic spectral models, saved in the 'models' directory.
    - Log files are saved in the 'logs' directory.
    - A random sample of generated models is plotted in an array of plots.
    - Grouped and aggregated data from IPAC files is stored in a Pandas DataFrame.

Author: Antonio Tutone
Date: 21/11/2023
"""

# Import necessary libraries
import datetime
import astropy.units as u
import numpy as np
from xspec import AllModels, AllData, Model, Plot
from pathlib import Path
from scipy.stats import qmc
from astropy.io import ascii
from astropy.table import Table

# Import custom modules
from modules.utils import plot_random_sample, process_ipac_files
from modules.logging_config import logging_conf

# Set the size of the Dataset
N = 100

# Set up paths for logs and models
cwd = Path.cwd()
path_to_logs = Path(cwd / "logs")
path_to_logs.mkdir(parents=True, exist_ok=True)
path_to_models = Path(cwd / "models")
path_to_models.mkdir(parents=True, exist_ok=True)

# Get the current date and time
now = datetime.datetime.now()
# Format the current date and time as a string
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

# Set up log configuration and create a logger for the fit
logger = logging_conf(path_to_logs, f"models_creator_{timestamp}.log")

# Debug: Log the start of the script
logger.debug("Script started.")

# Create the model
model_name = "TBfeo*(rdblur*rfxconv*comptb + diskbb + comptb)"
model = Model(model_name)
logger.debug(f"Model name: {model_name}")
logger.debug(f"Model used: {model.componentNames}")

# Changing default frozen parameters to unfrozen
model.rdblur.Betor10.frozen = False
model.rdblur.Rout_M.frozen = False
model.rfxconv.Fe_abund.frozen = False

# Collect the relevant parameter (the ones not frozen or linked)
unlink_params = model.nParameters - len(model.comptb_6.parameterNames)
relevant_par = []
for n_par in range(1, unlink_params + 1):
    if not model(n_par).frozen:
        relevant_par.append(n_par)

# Create a Latin Hypercube sampler for model parameters
sampler = qmc.LatinHypercube(d=len(relevant_par))
sample = sampler.random(n=N)

# Extract lower and upper bounds, and parameter names for scaling
l_bounds, u_bounds, par_names = [], [], []

for n_par in relevant_par:
    l_bounds.append(model(n_par).values[3]) #bot
    u_bounds.append(model(n_par).values[4]) #top
    par_names.append(model(n_par).name)

logger.debug(f"Components in the model: {par_names}")
# Scale the sample to fit parameter bounds
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

# Iterate through scaled samples to set up and save models
for idx, params in enumerate(sample_scaled):
    # Debug: Log parameters for the current iteration
    logger.debug(f"Step number: {idx}/{len(sample_scaled)}")
    logger.debug(f"Current parameters: {params}")

    # Clear existing XSPEC models and data
    AllModels.clear()
    AllData.clear()
    AllData.dummyrsp(0.1 ,100.)

    # Initialize the model with the scaled parameters
    m = Model(model_name, setPars={relevant_par[i]:params[i] for i in range(len(relevant_par))})

    # Changing default frozen parameters to unfrozen
    m.rdblur.Betor10.frozen = False
    m.rdblur.Rout_M.frozen = False
    m.rfxconv.Fe_abund.frozen = False

    # Linking comptb_6 (refletion) parameters to comptb (comptb)
    start = 23  # Number of the first parameter of comptb_6
    for i in range(start, start + len(m.comptb_6.parameterNames)):
        m(i).link = m(i-18) # 18 is the separation between comptb and comptb_6

    # Add the model to the spectral analysis system
    AllModels.setPars(m)

    # Set up the energy range of interest for plotting
    Plot.device = "/null"
    Plot.xAxis = "keV"
    Plot.show()
    Plot('model')
    energy = Plot.x()
    flux = Plot.model()

    # Smooth the data by averaging every 'm' consecutive points
    energy = np.array(energy)
    flux = np.array(flux)
    m = 10
    energy = energy.reshape(-1, m).mean(axis=1)
    flux = flux.reshape(-1, m).mean(axis=1)

    # Create a table with energy and flux
    table = Table([energy * u.keV, flux / (u.cm**2 * u.s * u.keV)], names=['Energy', 'Flux'])

    # Add comments to the table metadata
    table.meta['comments'] = [
        'Parameters used to generate the data:',
        f'Parameter names: {", ".join(par_names)}',
        f'Values: {", ".join(map(lambda x: f"{x:.9g}", params))}'
    ]

    # Create the file name based on parameter values
    file_name = f"model_{idx:04d}_params" + "_".join([f"{format(param, '.1e')}" for param in params]) + ".ipac"

    # Save the table in Ipac format with the created file name
    ascii.write(table, path_to_models / file_name, format='ipac', overwrite=True)

# Debug: Log the end of the script
logger.debug("Script completed.")

# Plot random sample of generated models
plot_random_sample(path_to_models, n_plots_per_row=3)

# Process the generated models and display grouped data
res = process_ipac_files(path_to_models)