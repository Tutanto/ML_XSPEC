"""
Updated K-Fold Cross-Validation Training Script for an ANN Model

This enhanced script trains an Artificial Neural Network (ANN) model using K-fold cross-validation, focusing on maximizing the model's robustness and generalizability across various data segments. 
Specifically tailored for astronomical datasets, it can be easily adapted for other types of data. 
Featuring comprehensive logging and TensorBoard integration, the script offers detailed insights into the training process and model performance.

Dependencies:
- numpy for array operations,
- tensorflow for building and training the ANN model,
- sklearn for K-fold cross-validation,
- Custom modules including modules.network for ANN architecture and modules.variables for path management,
- logging_config for structured logging.

Enhancements and Features:
- Explicit Logging: Initializes a logging process, recording the script's execution and model performance metrics, thereby enhancing traceability and debugging capabilities.
- Configurable Model Parameters: Allows customization of the ANN model's structure, including the number of neurons, layers, and dropout rate, alongside training configurations such as the number of epochs and batch size.
- Environment Setup: Prepares the operating environment by managing GPU visibility settings to ensure optimal hardware utilization.
- Advanced Data Handling: In addition to loading preprocessed data, the script incorporates detailed logging at each significant step, from data loading through to model evaluation, ensuring transparency and reproducibility.

Usage Steps:
1. Confirm the dataset's placement in the specified directory and validate the path configurations in `modules.variables`.
2. Customize the cross-validation folds, random seed, and model parameters according to the dataset characteristics and training requirements.
3. Execute the script in a compatible Python environment where all dependencies are resolved. The script manages data partitioning, model instantiation, training, and logging.
4. Utilize TensorBoard to visually track and analyze the training and validation metrics, facilitating informed model tuning and comparison.

Output Artifacts:
- A well-trained ANN model, saved under the 'path_to_results' directory, fine-tuned for high performance on diverse data segments.
- Detailed logs of training progress and model evaluation metrics, available for review and analysis.
- A comprehensive training history, aggregated across all folds, saved in JSON format, offering a holistic view of the model's learning curve and performance benchmarks.

Configuration Example:
- K-Fold Cross-Validation: k=5, seed=42 for deterministic shuffling.
- Model Architecture: neurons=256, layers=6, dropout=0.3, tailored for complex pattern recognition tasks.
- Training Setup: epochs=50, batch_size=50, balancing training depth with computational efficiency.

Note: Model parameters and training configurations are adjustable to cater to specific requirements of the dataset or computational constraints.
"""

import json
import datetime
import numpy as np

from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from modules.network import ANN_model
from logging_config import logging_conf

from modules.variables import (
    path_to_logs,
    path_to_data,
    path_to_results
)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

data = 'models_100k'
# Parameters for k-fold validation
k = 5  # Number of folds
seed = 42  # Random seed for reproducibility
neurons = 32
layers = 6
epochs = 250

# Initialize the logging process with timestamp
t_start = datetime.datetime.now()
timestamp = t_start.strftime("%Y-%m-%d_%H-%M-%S")
path_to_logs.mkdir(parents=True, exist_ok=True)
logger = logging_conf(path_to_logs, f"ANN_{timestamp}.log")
logger.debug("Script started.")

path_to_data = path_to_data / data
path_to_results.mkdir(parents=True, exist_ok=True)
log_dir = path_to_logs / 'fit'
log_dir.mkdir(parents=True, exist_ok=True)

# Load the datasets
logger.debug("Loading the datasets...")
X = np.load(path_to_data / 'Inp_norm.npy', allow_pickle=True)
Y = np.load(path_to_data / 'Out_norm.npy', allow_pickle=True)

# Initialize KFold
kf = KFold(n_splits=k, shuffle=True, random_state=seed)
histories = {'mean_absolute_error':[], 'mean_squared_error':[], 'r_squared':[],  
             'loss':[], 'val_mean_absolute_error':[], 'val_mean_squared_error': [], 
             'val_r_squared':[], 'val_loss':[]}

# Early Stopping Callback configuration
early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

# Loop over each fold
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    logger.debug(f"Training on fold {fold+1}/{k}...")

    # Splitting data into training and validation sets for this fold
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = Y[train_index], Y[val_index]
        
    # Create a TensorBoard instance with log directory
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir / now, histogram_freq=1)

    logger.debug("Creating the model...")
    model = ANN_model(X_train.shape[1], y_train.shape[1], neurons=neurons, hidden=layers, dropout=0.3)
    logger.debug("Model created")

    # Train the model
    logger.debug(f"Neurons: {neurons}")
    logger.debug(f"Hidden Layers: {layers}")
    logger.debug(f"Training for {epochs} epochs...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val), 
        epochs=epochs, batch_size=50,
        callbacks=[tensorboard_callback, early_stopping],
        verbose=1
    )

    logger.debug(f"End of fold {fold+1}")

    # Evaluate the model on the test set
    test_loss, test_mse, test_mae, test_r2 = model.evaluate(X_val, y_val)
    logger.debug(f"Test MAE: {test_mae}, Test MSE: {test_mse}, Test R2: {test_r2}")
    logger.debug(f'Score for fold {fold+1}: {model.metrics_names[0]} of {test_loss}')

    histories['mean_absolute_error'].append(history.history['mean_absolute_error'])
    histories['mean_squared_error'].append(history.history['mean_squared_error'])
    histories['r_squared'].append(history.history['r_squared'])
    histories['loss'].append(history.history['loss'])
    histories['val_mean_absolute_error'].append(history.history['val_mean_absolute_error'])
    histories['val_mean_squared_error'].append(history.history['val_mean_squared_error'])
    histories['val_r_squared'].append(history.history['val_r_squared'])    
    histories['val_loss'].append(history.history['val_loss'])
    logger.debug("History updated")

logger.debug("End of training!")
model.save(path_to_results / 'ANN_model_norm.h5')
logger.debug(f"Model saved in: {path_to_results}")

# Save merged history
history_filename = path_to_results / 'ANN_training_history_norm.json'
with open(history_filename, 'w') as f:
    json.dump(histories, f)
logger.debug(f"History saved: {history_filename}")
