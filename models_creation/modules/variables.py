from pathlib import Path

cwd = Path.cwd()
root_dir = cwd.parents[0]

# Set up paths for logs and models
path_to_logs = Path(root_dir / "logs")
path_to_models = Path(root_dir / "models")
path_to_samples = Path(root_dir / "samples")
path_to_batches = Path(root_dir / "batches")
path_to_all_models = Path(root_dir / "all_models")
path_to_checkpoints = Path(root_dir / "checkpoints")