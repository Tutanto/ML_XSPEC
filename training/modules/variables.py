from pathlib import Path

cwd = Path.cwd()
root_dir = cwd.parent

# Set up paths for logs and models
path_to_logs = cwd / "logs"
path_to_plots = cwd / "plots"
path_to_data = root_dir / "data"
path_to_batches = root_dir / "batches"
path_to_results = root_dir / "results"
path_to_samples = root_dir / "samples"
path_to_all_models = root_dir / "all_models"