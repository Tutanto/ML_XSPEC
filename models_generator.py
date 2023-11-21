"""
Script: model_generation_script.py

Description:
    This script generates and saves synthetic spectral models using the XSPEC library. It creates a Latin Hypercube
    sampler to sample parameters for a diskline model and then scales the samples to fit the parameter bounds.
    The script iterates through the scaled samples, sets up XSPEC models with the sampled parameters, and saves
    the generated models as IPAC format tables.

Dependencies:
    - Python 3.x
    - XSPEC (XSPEC models and data fitting library)
    - Astropy (for handling units and table creation)
    - SciPy (for Latin Hypercube sampling)
    - Modules (custom modules: utils, logging_config)

Usage:
    1. Ensure that all dependencies are installed.
    2. Run the script.

Outputs:
    - IPAC format tables containing synthetic spectral models, saved in the 'models' directory.
    - Log files are saved in the 'logs' directory.

Author: Antonio Tutone
Date: 20/11/2023
"""

# Import necessary libraries
import datetime
import astropy.units as u
from xspec import AllModels, AllData, Model, Plot
from pathlib import Path
from scipy.stats import qmc
from astropy.io import ascii
from astropy.table import Table

# Import custom modules
from modules.utils import plot_random_sample
from modules.logging_config import logging_conf

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

# Create a diskline model
model = Model("diskline")
logger.debug(f"Model used: {model.componentNames}")

# Create a Latin Hypercube sampler for model parameters
sampler = qmc.LatinHypercube(d=model.nParameters)
sample = sampler.random(n=10)

# Extract lower and upper bounds, and parameter names for scaling
l_bounds, u_bounds, par_names = [], [], []
for n_par in range(1, model.nParameters + 1):
    l_bounds.append(model(n_par).values[2])
    u_bounds.append(model(n_par).values[-1])
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
    AllData.dummyrsp(0.1 ,110.)

    # Initialize the diskline model with the scaled parameters
    m = Model("diskline", setPars=(list(params)))

    # Add the model to the spectral analysis system
    AllModels.setPars(m)

    # Set up the energy range of interest for plotting
    Plot.device = "/null"
    Plot.xAxis = "keV"
    Plot.show()
    Plot('model')
    energy = Plot.x()
    flux = Plot.model()

    # Create a table with energy and flux
    table = Table([energy * u.keV, flux / (u.cm**2 * u.s * u.keV)], names=['Energy', 'Flux'])

    # Add comments to the table metadata
    table.meta['comments'] = [
        'Parameters used to generate the data:',
        f'Parameter names: {", ".join(par_names)}',
        f'Values: {", ".join(map(lambda x: f"{x:.8g}", params))}'
    ]

    # Create the file name based on parameter values
    file_name = f"model_{idx:04d}_params" + "_".join([f"{format(param, '.1e')}" for param in params]) + ".ipac"

    # Save the table in Ipac format with the created file name
    ascii.write(table, path_to_models / file_name, format='ipac', overwrite=True)

# Debug: Log the end of the script
logger.debug("Script completed.")

# Plot random sample of generated models
plot_random_sample(path_to_models, n_plots_per_row=3)
