import time
import numpy as np
import matplotlib.pyplot as plt

from joblib import load
from tensorflow.keras.models import load_model
from xspec import FakeitSettings, AllModels, AllData, Model, Plot

from modules.network import r_squared

from modules.variables import (
    path_to_data,
    path_to_plots,
    path_to_results,
    par_original
)

n_spec = 1
# Define data
data = 'models_100k'
NN = 'GRU'
arch = '256x4'

path_to_plots.mkdir(parents=True, exist_ok=True)
path_to_files = path_to_data / 'files'
result_dir = path_to_results / NN / arch
model_file_path = result_dir / f'{NN}_model_norm.h5'
path_to_data_scaler = path_to_data / data

fs1 = FakeitSettings(response=path_to_files.__str__()+"/ni5050300117mpu7.rmf", arf=path_to_files.__str__()+"/ni5050300117mpu7.arf", exposure="1e5", fileName='test.fak')
times = []

for i in range(n_spec):
    start_time = time.time()
    # Clear existing XSPEC models and data
    AllModels.clear()
    AllData.clear()

    # Create the model
    model_name = "TBabs*(rdblur*rfxconv*comptb + diskbb + comptb)"
    true_model = Model(model_name)

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
    times.append(time.time() - start_time)


# Load scalers and normalize parameters if needed
if 'norm' in model_file_path.name:
    # Load the scalers
    loaded_scalers = {}
    parameters = np.zeros(par_original[:-1].shape[0])
    for i in range(par_original[:-1].shape[0]):
        loaded_scalers[i] = load(path_to_data_scaler / f'scaler_{i}.joblib')
    for i in range(par_original[:-1].shape[0]):        
        parameters[i] = loaded_scalers[i].transform(par_original[i].reshape(-1,1)).flatten()


# Load the saved scaler
flux_scaler = load(path_to_data_scaler / 'flux_scaler.joblib')
predicted = []
predicted_times = []

for i in range(n_spec):
    start_time = time.time()
    # Load the deep learning model
    model = load_model(model_file_path, custom_objects={'r_squared': r_squared})
    model_par = parameters
    y_pred = model.predict(model_par.reshape(-1, model_par.shape[0]), verbose=0)
    y_pred_d = flux_scaler.inverse_transform(y_pred)
    predicted_times.append(time.time() - start_time)

# Spectrum
plt.figure(figsize=(10, 6))
plt.errorbar(energy, y, yerr=yerr, fmt="o", color="cyan", ecolor='lightblue', elinewidth=3, capsize=0, label="XSPEC Model", alpha=0.5)
plt.plot(energy, y_pred_d[0], "^-r", label="ML Model", linewidth=2, markersize=7, alpha=0.5)
plt.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1.1, 1.05))
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("E (keV)", fontsize=14)
plt.ylabel("counts / (keV s)", fontsize=14)
plt.title("Comparison of XSPEC and ML Model Predictions", fontsize=16)
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.savefig(path_to_plots / "spectra.png")
    
# Histogram
plt.figure(figsize=(10, 6))
# Use the `histtype='stepfilled'` for a solid histogram with edges. `align='mid'` will align the bins to the center by default.
# By specifying different (but complementary) colors and adjusting the edge color, we improve the visibility of overlapping bins.
# Increase `bins` to a higher number for finer granularity if your data set is large enough to warrant it.
n_bins = max(len(set(times)), len(set(predicted_times)))  # Suggested dynamic bin size
plt.hist(times, bins=n_bins, alpha=0.7, label='Times XSPEC', color='blue', edgecolor='black', histtype='stepfilled')
plt.hist(predicted_times, bins=n_bins, alpha=0.7, label='Times ML', color='orange', edgecolor='black', histtype='stepfilled')
plt.yscale('log')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Times Distribution', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
plt.savefig(path_to_plots / "histogram.png")
plt.show()