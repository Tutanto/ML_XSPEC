import time
import emcee
import numpy as np
import matplotlib.pyplot as plt

from joblib import load
from multiprocessing import Pool
from scipy.optimize import minimize
from IPython.display import Math
from tensorflow.keras.models import load_model

from logging_config import logging_conf

from modules.parameters import Parameters
from modules.network import r_squared
from modules.variables import (
    params,
    par_original,
    path_to_mcmc,
    path_to_data, 
    path_to_logs,
    path_to_plots,
    path_to_results,
    path_to_data_points
)

def Model(x, par):
    x_grid = (np.array(x).reshape(-1, n_points).mean(axis=1)).tolist()
    y_pred = model.predict(par.reshape(-1, par.shape[0]), verbose=0)
    y_pred_d = flux_scaler.inverse_transform(y_pred)

    return np.interp(x, x_grid, y_pred_d[0])

def log_likelihood(theta):
    model_par = theta[:-1].copy()
    log_f = theta[-1]
    model = Model(xd, model_par)
    sigma2 = yderr**2 + model**2 * np.exp(2 * log_f)
#    sigma2 = yerr**2 + np.exp(2 * log_f)
    return -0.5 * np.sum((yd - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(params):
    # Assuming params is an instance of Parameters class

    # Automatically check each parameter in the params object
    for param_name in params.__dict__:
        # Skip checking for attributes that are not parameters (like 'ranges')
        if param_name == 'ranges':
            continue

        if not params.is_param_within_range(param_name):
            return -np.inf

    # If all parameters are within range, return 0.0 or another suitable value
    return 0.0

def log_probability(theta):
    params = Parameters()
    params.update_from_array(theta)
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf, None
    return lp + log_likelihood(theta), lp

# Set up log configuration and create a logger for the fit
path_to_logs.mkdir(exist_ok=True)
logger = logging_conf(path_to_logs, "mcmc.log")

# Debug: Log the start of the script
logger.debug("Script started.")

# Set up paths for logs and models
path_to_mcmc.mkdir(exist_ok=True)  # Create the directory if it doesn't exist
path_to_plots.mkdir(exist_ok=True)

n_points = 1
data = 'models_100k'
path_to_data_scaler = path_to_data / data
flux_scaler = load(path_to_data_scaler / 'flux_scaler.joblib') # Load the saved scaler
logger.debug(f"Reading Scalers from: {path_to_data_scaler}")

result_dir = path_to_results / 'ANN' / '256x4'
model_file_path = result_dir / 'ANN_model_norm.h5'
model = load_model(model_file_path, custom_objects={'r_squared': r_squared})
logger.debug(f"Reading Deep Learning model from: {model_file_path}")

xd = np.load(path_to_data_points / 'energy_true.npy', allow_pickle=True)
yd = np.load(path_to_data_points / 'y_true.npy', allow_pickle=True)
yderr = np.load(path_to_data_points / 'yerr.npy', allow_pickle=True)
logger.debug(f"Reading data points to fit from: {path_to_data_points}")

logger.debug(f"The true parameters are: {par_original}")

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = par_original + 0.1 * np.random.randn(14)
soln = minimize(nll, initial)
# Assuming soln is the result of your optimization process
params_ml = Parameters()
params_ml.update_from_array(soln.x)
par_ml = params_ml.to_array()

logger.debug("Maximum likelihood estimates:")
logger.debug(f"{params_ml}")

plt.errorbar(xd, yd, yerr=yderr, fmt=".c", capsize=0, alpha=0.02, label="True Points")
plt.plot(xd, Model(xd, par_ml[:-1]), ":r", label="ML")
plt.legend(fontsize=14)
plt.xlabel("x")
plt.ylabel("y")
# Save the last plot to the specified directory
plot_path = path_to_plots / "quick_fit.png"
plt.savefig(plot_path)
logger.debug(f"Quick fit plot saved to: {plot_path}")

init = par_original

pos = init + 1e-4 * np.random.randn(128, len(init))
nwalkers, ndim = pos.shape
max_n = 2500

# Set up the backend
# Don't forget to clear it in case the file already exists
filename = "mcmc.h5"
backend = emcee.backends.HDFBackend(path_to_mcmc / filename)
backend.reset(nwalkers, ndim)

start_time = time.time()

logger.debug('Start sampler')

with Pool(20) as pool:
    sampler = emcee.EnsembleSampler(
        nwalkers, 
        ndim, 
        log_probability, 
        moves=[
            (emcee.moves.DEMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
            ], 
        backend=backend, 
        blobs_dtype=float, 
        a=3.0)

    sampler.run_mcmc(pos, max_n, progress=True)

'''sampler = emcee.EnsembleSampler(
    nwalkers, 
    ndim, 
    log_probability, 
    #moves=[
    #    (emcee.moves.DEMove(), 0.8),
    #    (emcee.moves.DESnookerMove(), 0.2),
    #], 
    backend=backend, 
    blobs_dtype=float, 
    #a=5.0
    )
sampler.run_mcmc(pos, max_n, progress=True)
'''

logger.debug(f"--- {time.time() - start_time} seconds ---")

samples = sampler.get_chain()
names = [name for name in params.__dict__]
labels = names[:-1]

for i in range(ndim):
    mcmc = np.percentile(samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    logger.debug(Math(txt))
logger.debug("True values:")
logger.debug(params)