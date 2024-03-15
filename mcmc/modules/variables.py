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
    nH=0.257, 
    Betor10=-1.754, 
    Rin_M=11.68, 
    Incl=42.96, 
    rel_refl=-0.10, 
    Fe_abund=2.908, 
    log_xi=2.7793, 
    kTs=0.5148, 
    alpha=0.8595, 
    kTe=2.0501, 
    norm=1.79977, 
    Tin=0.3018, 
    norm_disk=np.log10(5.1879e5))

par_original = params.to_array()
