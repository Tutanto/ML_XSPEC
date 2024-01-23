# Import necessary libraries
import sys
import json
import datetime
import numpy as np
from xspec import AllModels, AllData, Model, Plot
from pathlib import Path

# Import custom modules
from modules.utils import (
    extract_number,
    read_last_successful_index,
    save_last_successful_index,
    )

sample_file_name = "split_300-399.npy"
cwd = Path.cwd()
path_to_models = Path(cwd / "models")
path_to_samples = Path(cwd / "samples")
sample_file_path = path_to_samples / sample_file_name

model_name = "TBabs*(rdblur*rfxconv*comptb + diskbb + comptb)"
sample_scaled = np.load(sample_file_path)
relevant_par =  np.load(path_to_samples / "relevant_par.npy")

# Invert the log10 of these components
log_index = [0, 2, 9, 12]
for i in range(sample_scaled.shape[0]):
    for j in log_index:
        sample_scaled[i, j] =  pow(10, sample_scaled[i, j])

# Clear existing XSPEC models and data
AllModels.clear()
AllData.clear()
AllData.dummyrsp(0.5, 20.)

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
    m(i).link = m(i-9) # 9 is the separation between comptb and comptb_6

#m.setPars({int(relevant_par[j]):sample_scaled[29][j] for j in range(len(relevant_par))})
## Add the model to the spectral analysis system and set parameters
#AllModels.setPars(m, {int(relevant_par[j]):sample_scaled[29][j] for j in range(len(relevant_par))})
AllModels.setPars(m, {int(relevant_par[j]):sample_scaled[29][j] for j in range(9)})

# Set up the energy range of interest for plotting
'''Plot.device = "/xs"
Plot.xAxis = "keV"
Plot.show()
Plot('model')
energy = Plot.x()
flux = Plot.model()'''