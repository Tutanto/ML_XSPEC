import corner
import emcee
import time
import numpy as np

from joblib import load
from pathlib import Path
from multiprocessing import Pool
from scipy.optimize import minimize
from IPython.display import display, Math
from tensorflow.keras.models import load_model

from modules.parameters import Parameters
from modules.network import r_squared
from modules.variables import path_to_data, path_to_mcmc, path_to_results

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

data = 'models_0.5-20_100k_smooth_2'

path_to_data_scaler = path_to_data / data
target_file_path = path_to_data / 'target'
path_to_mcmc.mkdir(exist_ok=True)  # Create the directory if it doesn't exist'

result_dir = path_to_results / 'GRU' / f'100k' / '256x5'
model_file_path = result_dir / 'GRU_model.h5'

xd = np.load(target_file_path / 'energy_true.npy', allow_pickle=True)
yd = np.load(target_file_path / 'y_true.npy', allow_pickle=True)
yderr = np.load(target_file_path / 'yerr.npy', allow_pickle=True)

model = load_model(model_file_path, custom_objects={'r_squared': r_squared})
# Load the saved scaler
flux_scaler = load(path_to_data / 'flux_scaler.joblib')

# The "true" parameters.
params = Parameters(
    nH=np.log10(1.0), 
    Betor10=-2, 
    Rin_M=10, 
    Incl=30, 
    rel_refl=-0.5, 
    Fe_abund=1, 
    log_xi=2, 
    kTs=1, 
    alpha=2, 
    kTe=np.log10(40), 
    norm=0.5, 
    Tin=1, 
    norm_disk=np.log10(1), 
    f_true=np.log10(0.2))

par_original = params.to_array()

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = par_original + 0.1 * np.random.randn(14)
soln = minimize(nll, initial)
# Assuming soln is the result of your optimization process
params_ml = Parameters()
params_ml.update_from_array(soln.x)
par_ml = params_ml.to_array()

print("Maximum likelihood estimates:")
print(params_ml)

plt.errorbar(xd, yd, yerr=yderr, fmt=".c", capsize=0, alpha=0.02)
plt.plot(xd, Model(xd, par_ml[:-1]), ":r", label="ML")
plt.legend(fontsize=14)
plt.xlabel("x")
plt.ylabel("y")

init = par_original

pos = init + 1e-1 * np.random.randn(256, len(init))
nwalkers, ndim = pos.shape
max_n = 5000

# Set up the backend
# Don't forget to clear it in case the file already exists
filename = "tutorial.h5"
backend = emcee.backends.HDFBackend(mcmc_path / filename)
backend.reset(nwalkers, ndim)

start_time = time.time()

print('start sampler')

'''with Pool() as pool:
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

    sampler.run_mcmc(pos, 1000, progress=True)'''

sampler = emcee.EnsembleSampler(
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


print("--- %s seconds ---" % (time.time() - start_time))