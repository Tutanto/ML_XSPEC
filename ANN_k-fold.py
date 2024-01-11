import os
import h5py
import datetime
import numpy as np
import pandas as pd
from joblib import dump
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import TensorBoard


from modules.utils import (
    process_json_files_batch, 
    combine_hdf5_files, 
    remove_uniform_columns
)

from modules.network import (
    calc_mean_std_per_epoch,
    plot_two_metrics,
    ANN_model
)

os.environ['CUDA_VISIBLE_DEVICES'] = ''

path_to_models = Path(Path.cwd() / 'all_models' / 'models_0.7-20_10k')
path_to_logs = Path(Path.cwd() / 'logs')
path_to_batches = Path(Path.cwd() / 'batches')
path_to_batches.mkdir(parents=True, exist_ok=True)
log_dir = path_to_logs / 'fit'
log_dir.mkdir(parents=True, exist_ok=True)

# Check if any 'batch_*.h5' files already exist
existing_batches = list(path_to_batches.glob('batch_*.h5'))
if not existing_batches:
    # Read the json models in batches
    for i, (flux, params) in enumerate(process_json_files_batch(path_to_models)):
        with h5py.File(path_to_batches / f'batch_{i}.h5', 'w') as hf:
            hf.create_dataset('flux', data=flux)
            hf.create_dataset('params', data=params)

# Put all the batches together
all_flux_values, all_parameters = combine_hdf5_files(path_to_batches)
# Convert y_train to a float data type
all_parameters = np.array(all_parameters, dtype=float)

# Use the function on your data
relevant_parameters, removed_columns = remove_uniform_columns(all_parameters)

print("Modified Data:")
print(relevant_parameters)
print("Removed Columns Indices:")
print(removed_columns)

# Applica MinMaxScaler per Ogni Colonna (Parametro)
scalers = {}
Y = np.zeros_like(relevant_parameters)

for i in range(relevant_parameters.shape[1]):  # Itera attraverso le colonne (parametri)
    scaler = MinMaxScaler(feature_range=(0, 1))
    Y[:, i] = scaler.fit_transform(relevant_parameters[:, i].reshape(-1, 1)).flatten()
    scalers[i] = scaler  # Memorizza il scaler per un uso futuro

# Save each Scaler to a separate file
for i, scaler in scalers.items():
    dump(scaler, log_dir / f'scaler_{i}.joblib')

# Normalize flux values
X_scaler = MinMaxScaler(feature_range=(0, 1))
X = X_scaler.fit_transform(all_flux_values.reshape(-1, all_flux_values.shape[-1])).reshape(all_flux_values.shape)
# Save the scaler
dump(X_scaler, log_dir / 'X_scaler.joblib')

# Find the indices of rows that contain NaN
indices_with_nan = np.any(np.isnan(X), axis=1)
# Count the rows to be removed
rows_to_remove = np.sum(indices_with_nan)
# Remove the rows that contain NaN
X = X[~indices_with_nan]
Y = Y[~indices_with_nan]
# Print the number of rows removed
print(f"Number of rows removed: {rows_to_remove}")

# Parameters for k-fold validation
k = 5  # Number of folds
seed = 42  # Random seed for reproducibility
# Initialize KFold
kf = KFold(n_splits=k, shuffle=True, random_state=seed)
histories = {'mean_absolute_error':[], 'mean_squared_error':[], 'r_squared':[],  
             'loss':[], 'val_mean_absolute_error':[], 'val_mean_squared_error': [], 
             'val_r_squared':[], 'val_loss':[]}

# Loop over each fold
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Training on fold {fold+1}/{k}...")

    # Splitting data into training and validation sets for this fold
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = Y[train_index], Y[val_index]
        
    # Create a TensorBoard instance with log directory
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir / now, histogram_freq=1)

    model = ANN_model(X_train.shape[1], y_train.shape[1])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val), 
        epochs=150, batch_size=100,
        callbacks=[tensorboard_callback],
        verbose=0
    )

    # Evaluate the model on the test set
    test_loss, test_mse, test_mae, test_r2 = model.evaluate(X_val, y_val)
    print(f"Test MAE: {test_mae}, Test MSE: {test_mse}, Test R2: {test_r2}")
    print(f'Score for fold {fold+1}: {model.metrics_names[0]} of {test_loss}')

    histories['mean_absolute_error'].append(history.history['mean_absolute_error'])
    histories['mean_squared_error'].append(history.history['mean_squared_error'])
    histories['r_squared'].append(history.history['r_squared'])
    histories['loss'].append(history.history['loss'])
    histories['val_mean_absolute_error'].append(history.history['val_mean_absolute_error'])
    histories['val_mean_squared_error'].append(history.history['val_mean_squared_error'])
    histories['val_r_squared'].append(history.history['val_r_squared'])    
    histories['val_loss'].append(history.history['val_loss'])

model.save(log_dir / 'my_model.h5')
history_df = pd.DataFrame(histories)

# Plot loss and val_loss
mean_loss, std_loss = calc_mean_std_per_epoch(history_df['loss'])
mean_val_loss, std_val_loss = calc_mean_std_per_epoch(history_df['val_loss'])
epochs = range(len(mean_loss))
plot_two_metrics(epochs, mean_loss, std_loss, 'Loss', mean_val_loss, std_val_loss, 'Val Loss', 'Loss vs. Validation Loss')

# Plot mean_absolute_error and val_mean_absolute_error
mean_mae, std_mae = calc_mean_std_per_epoch(history_df['mean_absolute_error'])
mean_val_mae, std_val_mae = calc_mean_std_per_epoch(history_df['val_mean_absolute_error'])
epochs = range(len(mean_mae))
plot_two_metrics(epochs, mean_mae, std_mae, 'MAE', mean_val_mae, std_val_mae, 'Val MAE', 'MAE vs. Validation MAE')

# Plot r2 and val_r2
mean_r2, std_r2 = calc_mean_std_per_epoch(history_df['r_squared'])
mean_val_r2, std_val_r2 = calc_mean_std_per_epoch(history_df['val_r_squared'])
epochs = range(len(mean_r2))
plot_two_metrics(epochs, mean_r2, std_r2, 'R2', mean_val_r2, std_val_r2, 'Val R2', 'R2 vs. Validation R2')