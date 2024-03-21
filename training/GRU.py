"""
Training and Evaluation Script for a GRU-based Neural Network Model

This script is responsible for training a Gated Recurrent Unit (GRU) model on a dataset of astronomical observations. 
The script supports both loading a pre-existing model and training a new model from scratch. 
It handles data preprocessing, model training with TensorBoard callbacks for visualization, and model evaluation. 
The training process, along with its configurations, is logged for reproducibility and monitoring.

Dependencies:
- External Libraries: numpy, tensorflow, sklearn
- Custom Modules: modules.network for model architecture and r_squared metric, logging_config for logging setup, modules.variables for path management

Features:
- Data Handling: Checks for the presence of preprocessed data and a pre-existing model. 
    If found, it loads them for use; otherwise, it preprocesses the raw data.
- Model Training: Trains a GRU model with specified configurations (number of neurons, layers, and epochs). 
    Training progress is logged, and TensorBoard is used for real-time monitoring.
- Model Saving: Saves the trained model and its training history for future reference and evaluation.
- Model Evaluation: Evaluates the model on a test set and logs key performance metrics.

Usage:
1. Ensure that the dataset is available in the specified directory and that `modules.variables` paths are correctly set up.
2. Configure the model parameters (neurons, layers, epochs) as desired.
3. Run the script in an environment where all dependencies are installed. 
    It will automatically handle data loading, model training or loading, and model evaluation.
4. Use TensorBoard to monitor training progress by pointing it to the log directory.

Output:
- The script outputs a trained GRU model, saved in the 'path_to_results' directory.
- Training and validation performance metrics are logged and saved in a JSON file for further analysis.
- The performance of the model on the test set is evaluated and logged, providing insights into its generalization capability.

Example Configuration:
- neurons: 256 (number of neurons in each GRU layer)
- layers: 4 (number of GRU layers in the model)
- epochs: 50 (number of training epochs)

Note: Adjust the 'neurons', 'layers', and 'epochs' variables as needed to optimize the model's performance for your specific dataset.
"""

import json
import datetime
import shutil
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from modules.network import r_squared, GRU_model
from logging_config import logging_conf

from modules.variables import (
    path_to_logs,
    path_to_data,
    path_to_results
)

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Initialize the logging process with timestamp
t_start = datetime.datetime.now()
timestamp = t_start.strftime("%Y-%m-%d_%H-%M-%S")
path_to_logs.mkdir(parents=True, exist_ok=True)
logger = logging_conf(path_to_logs, f"GRU_{timestamp}.log")
logger.debug("Script started.")

data = 'models_100k'
neurons = 64
layers = 4
epochs = 50
batch_size = 16
path_to_file_data = path_to_data / data
input_file = path_to_file_data / 'Inp_norm.npy'
output_file = path_to_file_data / 'Out_norm.npy'

logger.debug(f"Data: {path_to_file_data}")
log_dir = path_to_logs / 'fit'
log_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = path_to_results / 'training'
checkpoint_path = checkpoint_dir / "cp-{epoch:04d}.h5"
# Define the path for the model file
path_to_results.mkdir(parents=True, exist_ok=True)
if 'norm' in input_file.name:
    model_file_path = path_to_results / 'GRU_model_norm.h5'
else:
    model_file_path = path_to_results / 'GRU_model.h5'

# File paths for the saved datasets
X_train_file = path_to_file_data / 'X_train_par.npy'
y_train_file = path_to_file_data / 'y_train_flux.npy'
X_val_file = path_to_file_data / 'X_val_par.npy'
y_val_file = path_to_file_data / 'y_val_flux.npy'
X_test_file = path_to_file_data / 'X_test_par.npy'
y_test_file = path_to_file_data / 'y_test_flux.npy'

# Check if the datasets files exist
if X_train_file.is_file() and (model_file_path.is_file() or checkpoint_dir.is_dir()):
    # Load the datasets
    logger.debug("Loading the datasets...")
    # Load the datasets
    X_train_par = np.load(X_train_file, allow_pickle=True)
    y_train_flux = np.load(y_train_file, allow_pickle=True)
    X_val_par = np.load(X_val_file, allow_pickle=True)
    y_val_flux = np.load(y_val_file, allow_pickle=True)
    X_test_par = np.load(X_test_file, allow_pickle=True)
    y_test_flux = np.load(y_test_file, allow_pickle=True)
    if model_file_path.is_file():
        logger.debug("Saved model found!")
        logger.debug("Loading the saved model...")
        model = load_model(model_file_path, custom_objects={'r_squared': r_squared})
        logger.debug(f"Model loaded: {model_file_path}")
    else:
        logger.debug("Loading the latest checkpoint...")
        srt = sorted(checkpoint_dir.glob('*')) 
        latest = srt[-1]
        logger.debug(f"Checkpoint loaded: {latest}")
        model = load_model(latest, custom_objects={'r_squared': r_squared})
        logger.debug("Model re-loaded")
else:
    logger.debug("No saved model found. Using a new model...")
    # Load the datasets
    X = np.load(input_file, allow_pickle=True)
    Y = np.load(output_file, allow_pickle=True)
    logger.debug("Loaded input and output")
    # Split the data into training, validation, and test sets
    X_train_par, X_temp_par, y_train_flux, y_temp_flux = train_test_split(
        X, 
        Y, 
        test_size=0.3, random_state=42
        )
    X_val_par, X_test_par, y_val_flux, y_test_flux = train_test_split(
        X_temp_par, 
        y_temp_flux, 
        test_size=0.5, random_state=42
        )

    # Save the datasets
    np.save(X_train_file, X_train_par)
    np.save(y_train_file, y_train_flux)
    np.save(X_val_file, X_val_par)
    np.save(y_val_file, y_val_flux)
    np.save(X_test_file, X_test_par)
    np.save(y_test_file, y_test_flux)
    logger.debug(f"Datasets saved in: {path_to_file_data}")

    # Define the neural network model
    logger.debug("Creating the model...")
    model = GRU_model(X_train_par.shape[1], y_train_flux.shape[1], neurons=neurons, hidden=layers, learning_rate=1.e-3)
    logger.debug("Model created")
    
# Create a TensorBoard instance with log directory
tensorboard_callback = TensorBoard(log_dir=log_dir / f"log_{t_start}", histogram_freq=1)

# Calculate the number of batches per epoch
n_batches = len(X_test_par[0]) / batch_size
n_batches = int(np.ceil(n_batches))    # round up the number of batches to the nearest whole integer

# Create a callback that saves the model every 5 epochs
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path.as_posix(), 
    verbose=0, 
    save_weights_only=False,
    save_freq=5*n_batches)

# Train the model
logger.debug(f"Neurons: {neurons}")
logger.debug(f"Hidden Layers: {layers}")
logger.debug(f"Training for {epochs} epochs...")
new_history = model.fit(
    X_train_par, y_train_flux,
    validation_data=(X_val_par, y_val_flux), 
    epochs=epochs, batch_size=batch_size,
    callbacks=[cp_callback, tensorboard_callback],
    verbose=1
).history
logger.debug("End of training!")

# Save (or update) the model
model.save(model_file_path)
logger.debug(f"Model saved in: {model_file_path}")
# Remove the Checkpoints
shutil.rmtree(checkpoint_dir)
logger.debug(f"Checkpoints removed")

# Load existing history if it exists
if 'norm' in input_file.name:
    history_filename = path_to_results / 'GRU_training_history_norm.json'
else:
    history_filename = path_to_results / 'GRU_training_history.json'
if history_filename.exists():
    with open(history_filename, 'r') as f:
        existing_history = json.load(f)
else:
    existing_history = {}

# Merge new history with existing history
for key in new_history:
    if key in existing_history:
        existing_history[key].extend(new_history[key])
    else:
        existing_history[key] = new_history[key]

# Save merged history
with open(history_filename, 'w') as f:
    json.dump(existing_history, f)
logger.debug(f"History saved: {history_filename}")

# Evaluate the model on the test set
test_loss, test_mae, test_mse, test_r2 = model.evaluate(X_test_par, y_test_flux)
logger.debug(f"Test MAE: {test_mae}, Test MSE: {test_mse}, Test R2: {test_r2}")
logger.debug(f'Score: {model.metrics_names[0]} of {test_loss}')
