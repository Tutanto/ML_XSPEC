"""
Training History Visualization Script for Machine Learning Models

This script is designed to visualize the training history of machine learning models, specifically focusing on GRU (Gated Recurrent Unit) and ANN (Artificial Neural Network) models. 
It loads the training history from a JSON file, skips the initial epochs to focus on the later stages of training, and generates a set of plots to compare various performance metrics such as loss, mean absolute error (MAE), mean squared error (MSE), and R-squared (R2) values between the training and validation datasets.

Dependencies:
- json for loading the training history from a file,
- pandas for data manipulation,
- matplotlib for plotting,
- logging_config for logging setup and execution tracking,
- Custom modules including modules.network for calculation and plotting utilities, and modules.variables for path management.

Features:
- Flexible Data Loading: Capable of handling training history data for different models (GRU, ANN) stored in JSON format.
- Dynamic Plotting: Generates comparative plots for key metrics across training and validation phases, allowing for a detailed performance assessment.
- Configurable Plot Skipping: Ability to skip a specified number of initial epochs to focus on the more relevant later stages of training.
- Robust Logging: Utilizes a custom logging configuration to log script execution details, enhancing traceability and debugging.

Usage:
1. Ensure the training history JSON file exists in the specified directory structure as set in `modules.variables`.
2. Update the script's model type and name variables to match the target training history file.
3. Run the script in a compatible Python environment. It will automatically load the training history, generate plots for the specified metrics, and save the resulting figures.
4. Review the generated plots in the specified output directory to assess model performance and training dynamics.

Output:
- The script produces a multi-panel figure containing comparative plots of loss, MAE, MSE, and R2 metrics for the specified model, highlighting the differences between training and validation datasets.
- The generated figure is saved to the specified directory, allowing for easy access and further analysis.

Example Configuration:
- Model type: 'GRU' or 'ANN' (depending on the model being analyzed)
- Model name: Model identifier, e.g., '256x5' for a model with 256 neurons and 5 layers
- Skip: Number of initial epochs to skip in the visualization, e.g., 350

Note: The script's flexibility allows for easy adaptation to visualize the training history of other machine learning models by modifying the data loading and plotting sections accordingly.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from logging_config import logging_conf

from modules.network import (
    calc_mean_std_per_epoch,
    plot_two_metrics
)

from modules.variables import (
    path_to_logs,
    path_to_plots,
    path_to_results
)

path_to_logs.mkdir(parents=True, exist_ok=True)
path_to_plots.mkdir(parents=True, exist_ok=True)

# Initialize the logging process with timestamp
logger = logging_conf(path_to_logs, f"plot_history.log")
logger.debug("Script started.")

model = 'GRU'
model_name = '256x4'
model_dir = path_to_results / model
logger.debug(f"Models: {model_dir}")

# Load existing history if it exists
history_filename = model_dir / model_name / f'{model}_training_history_norm.json'
if history_filename.exists():
    with open(history_filename, 'r') as f:
        existing_history = json.load(f)
    logger.debug(f'Loaded history: {history_filename}')
else:
    logger.debug('No history')

skip = 50
history_df = pd.DataFrame(existing_history)
# Create a figure with subplots arranged in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
if 'GRU' in model:
    
    # Plot 1: Loss and Val Loss
    axs[0, 0].plot(history_df.loc[skip:, 'loss'], label='Loss')
    axs[0, 0].plot(history_df.loc[skip:, 'val_loss'], label='Val Loss')
    axs[0, 0].set_title('Loss vs. Val Loss')
    axs[0, 0].legend()

    # Plot 2: Mean Absolute Error
    axs[0, 1].plot(history_df.loc[skip:, 'mean_absolute_error'], label='MAE')
    axs[0, 1].plot(history_df.loc[skip:, 'val_mean_absolute_error'], label='Val MAE')
    axs[0, 1].set_title('MAE vs. Val MAE')
    axs[0, 1].legend()

    # Plot 3: Mean Squared Error
    axs[1, 0].plot(history_df.loc[skip:, 'mean_squared_error'], label='MSE')
    axs[1, 0].plot(history_df.loc[skip:, 'val_mean_squared_error'], label='Val MSE')
    axs[1, 0].set_title('MSE vs. Val MSE')
    axs[1, 0].legend()

    # Plot 4: R-Squared
    axs[1, 1].plot(history_df.loc[skip:, 'r_squared'], label='R2')
    axs[1, 1].plot(history_df.loc[skip:, 'val_r_squared'], label='Val R2')
    axs[1, 1].set_title('R2 vs. Val R2')
    axs[1, 1].legend()
    
    logger.debug("Minimum validation loss: {}".format(history_df['val_loss'].min()))

elif 'ANN' in model:
    # Plot loss and val_loss
    mean_loss, std_loss = calc_mean_std_per_epoch(history_df['loss'])
    mean_val_loss, std_val_loss = calc_mean_std_per_epoch(history_df['val_loss'])
    epochs = range(len(mean_loss))
    plot_two_metrics(epochs[skip:], mean_loss[skip:], std_loss[skip:], 'Loss', mean_val_loss[skip:], std_val_loss[skip:], 'Val Loss', 'Loss vs. Validation Loss', axs[0, 0])

    # Plot mean_absolute_error and val_mean_absolute_error
    mean_mae, std_mae = calc_mean_std_per_epoch(history_df['mean_absolute_error'])
    mean_val_mae, std_val_mae = calc_mean_std_per_epoch(history_df['val_mean_absolute_error'])
    epochs = range(len(mean_mae))
    plot_two_metrics(epochs[skip:], mean_mae[skip:], std_mae[skip:], 'MAE', mean_val_mae[skip:], std_val_mae[skip:], 'Val MAE', 'MAE vs. Validation MAE', axs[0, 1])
    
    # Plot mean_squared_error and val_mean_squared_error
    mean_mse, std_mse = calc_mean_std_per_epoch(history_df['mean_squared_error'])
    mean_val_mse, std_val_mse = calc_mean_std_per_epoch(history_df['val_mean_squared_error'])
    epochs = range(len(mean_mse))
    plot_two_metrics(epochs[skip:], mean_mse[skip:], std_mse[skip:], 'MSE', mean_val_mse[skip:], std_val_mse[skip:], 'Val MSE', 'MSE vs. Validation MSE', axs[1, 0])

    # Plot r2 and val_r2
    mean_r2, std_r2 = calc_mean_std_per_epoch(history_df['r_squared'])
    mean_val_r2, std_val_r2 = calc_mean_std_per_epoch(history_df['val_r_squared'])
    epochs = range(len(mean_r2))
    plot_two_metrics(epochs[skip:], mean_r2[skip:], std_r2[skip:], 'R2', mean_val_r2[skip:], std_val_r2[skip:], 'Val R2', 'R2 vs. Validation R2', axs[1, 1])

# Adjust layout so that titles and labels do not overlap
plt.tight_layout()

# Save the entire figure
fig.savefig(path_to_plots / f'{model}_{model_name}_metrics_comparison.png')