import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt

from logging_config import logging_conf

from modules.variables import (
    params,
    par_original,
    path_to_mcmc, 
    path_to_logs,
    path_to_plots
)

logger = logging_conf(path_to_logs, "plot_chains.log")
# Debug: Log the start of the script
logger.debug("Script started.")

filename = "mcmc.h5"
reader = emcee.backends.HDFBackend(path_to_mcmc / filename)
logger.debug(f"Reading: {filename}")

tau = reader.get_autocorr_time(tol=0)
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

logger.debug("burn-in: {0}".format(burnin))
logger.debug("thin: {0}".format(thin))
logger.debug("flat chain shape: {0}".format(samples.shape))
logger.debug("flat log prob shape: {0}".format(log_prob_samples.shape))
logger.debug("flat log prior shape: {0}".format(log_prior_samples.shape))

all_samples = np.concatenate(
    (samples, 
     log_prob_samples[:, None], 
     log_prior_samples[:, None]
     ), 
    axis=1
)

names = [name for name in params.__dict__]
labels = names[:-1]
#labels += ["log prob", "log prior"]

fig = corner.corner(
    samples, labels=labels, truths=par_original,
)

fig.savefig(path_to_plots / 'corner.png')
logger.debug(f"Corner plot saved in: {path_to_plots}")

fig, axes = plt.subplots(len(par_original), figsize=(20, 25), sharex=True)
for i in range(len(par_original)):
    ax = axes[i]
    ax.plot(samples[:, :, i], "c", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
plt.show()