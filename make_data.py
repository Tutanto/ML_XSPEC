import numpy as np
import matplotlib.pyplot as plt
from  pathlib import Path
from matplotlib import rcParams

from xspec import AllModels, AllData, Model, Plot

# Clear existing XSPEC models and data
AllModels.clear()
AllData.clear()
AllData.dummyrsp(0.5 ,20.)

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

true_model.comptb.kTe.values = 40
true_model.comptb.kTs.values = 2

# Linking the parameters
true_model.rfxconv.cosIncl.link = "COSD(5)"


start = 20  # Number of the first parameter of comptb_6
for i in range(start, start + len(true_model.comptb_6.parameterNames)):
    true_model(i).link = str(i-9) # 9 is the separation between comptb and comptb_6

# Collect the relevant parameter (the ones not frozen or linked)
relevant_par = []
relevant_val = []
for n_par in range(1, true_model.nParameters + 1):
    if not true_model(n_par).frozen and not true_model(n_par).link:
        relevant_val.append(true_model(n_par).values[0])
        relevant_par.append(n_par)

true_model.show()

# Set up the energy range of interest for plotting
Plot.device = "/null"
Plot.xAxis = "keV"
Plot.show()
Plot('model')
energy = Plot.x()
true_flux = Plot.model()

f_true = 0.1
y = true_flux
# Generate some synthetic data from the model.
yerr = 0.01 + 0.05 * np.random.rand(len(true_flux))
y += np.abs([f_true * true for true in true_flux]) * np.random.rand(len(true_flux))
y += yerr * np.random.rand(len(true_flux))

plt.errorbar(energy, y, yerr=yerr, fmt=".c", capsize=0)
plt.plot(energy,true_flux, "c", alpha=0.3, lw=3)
plt.ylabel("Flux(1 / keV cm^-2 s)")
plt.xlabel("Energy (KeV)")