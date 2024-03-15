import numpy as np
from  pathlib import Path
import matplotlib.pyplot as plt

from xspec import FakeitSettings, AllModels, AllData, Model, Plot

from logging_config import logging_conf

from modules.variables import (
    path_to_data,
    path_to_logs,
    path_to_plots
)

path_to_logs.mkdir(parents=True, exist_ok=True)
path_to_plots.mkdir(parents=True, exist_ok=True)
path_to_target = path_to_data / 'target'
path_to_target.mkdir(parents=True, exist_ok=True)     # Create the data directory if it doesn't exist
path_to_files = path_to_data / 'files'

# Initialize the logging process
logger = logging_conf(path_to_logs, "plot_history.log")
logger.debug("Script started.")

# Clear existing XSPEC models and data
AllModels.clear()
AllData.clear()

# Create the model
fs1 = FakeitSettings(response=path_to_files.__str__()+"/ni5050300117mpu7.rmf", arf=path_to_files.__str__()+"/ni5050300117mpu7.arf", exposure="1e5", fileName='test.fak')
model_name = "TBabs*(rdblur*rfxconv*comptb + diskbb + comptb)"
true_model = Model(model_name)
logger.debug(f"Model name: {model_name}")
logger.debug(f"Model used: {true_model.componentNames}")


# Changing default frozen parameters to unfrozen
true_model.rdblur.Betor10.frozen = False
true_model.rdblur.Rout_M.frozen = True
true_model.rdblur.Rin_M.frozen = False
true_model.rfxconv.Fe_abund.frozen = False
true_model.comptb.gamma.frozen = True
true_model.comptb.delta.frozen = True
true_model.comptb.log_A.frozen = True

true_model.rdblur.Rout_M.values = 1000
true_model.comptb.delta.values = 0
true_model.comptb.log_A.values = 8

true_model.TBabs.nH.values = 0.257
true_model.rdblur.Betor10.values = -1.754
true_model.rdblur.Rin_M.values = 11.68
true_model.rdblur.Incl.values = 42.96
true_model.rfxconv.rel_refl.values = -0.10
true_model.rfxconv.Fe_abund.values = 2.908
true_model.rfxconv.log_xi.values = 2.7793
true_model.comptb.kTe.values = 2.0501
true_model.comptb.kTs.values = 0.5148
true_model.comptb.alpha.values = 0.8595
true_model.comptb.norm.values = 1.79977
true_model.diskbb.Tin.values = 0.3018
true_model.diskbb.norm.values = 5.1879e5

# Linking the parameters
true_model.rfxconv.cosIncl.link = "COSD(5)"

start = 20  # Number of the first parameter of comptb_6
for i in range(start, start + len(true_model.comptb_6.parameterNames)):
    true_model(i).link = str(i-9) # 9 is the separation between comptb and comptb_6

logger.debug(true_model.show())
AllModels.setPars(true_model)
AllData.fakeit(1, fs1)
AllData.ignore('**-0.3')
AllData.ignore('10.-**')

# Set up for plotting
Plot.device = "/null"
Plot.xAxis = "keV"
Plot.show()
Plot('data')
energy = Plot.x()
energy_err = Plot.xErr()
y = Plot.y()
yerr = Plot.yErr()

plt.errorbar(energy, y, yerr=yerr, fmt=".c", capsize=0)
plt.ylabel("Flux [counts / (keV  s)]")
plt.xlabel("Energy (KeV)")
plt.savefig(path_to_plots / 'plot_target_model.png')
logger.debug(f'Plot saved in {path_to_plots}')

# Saving the data to disk
np.save(path_to_target / 'energy_true.npy', energy)
np.save(path_to_target / 'energy_error.npy', energy_err)
np.save(path_to_target / 'y_true.npy', y)
np.save(path_to_target / 'yerr.npy', yerr)
logger.debug(f'File saved in {path_to_target}')