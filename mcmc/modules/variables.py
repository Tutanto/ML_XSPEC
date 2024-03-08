import numpy as np
from pathlib import Path
from modules.parameters import Parameters

cwd = Path.cwd()
root_dir = cwd.parent

# Set up paths for logs and models
path_to_logs = cwd / "logs"
path_to_plots = cwd / "plots"
path_to_data = root_dir / "data"
path_to_mcmc = cwd / "mcmc_result"
path_to_data_points = cwd / "data"
path_to_batches = root_dir / "batches"
path_to_results = root_dir / "results"
path_to_samples = root_dir / "samples"
path_to_all_models = root_dir / "all_models"

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
