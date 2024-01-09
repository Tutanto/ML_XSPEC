import os
import h5py
import datetime
import numpy as np
import pandas as pd
from joblib import dump
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import backend as K
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

path_to_models = Path(Path.cwd() / 'models_0.1-100')
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

# Normalize flux values
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X = X_scaler.fit_transform(all_flux_values.reshape(-1, all_flux_values.shape[-1])).reshape(all_flux_values.shape)
Y = Y_scaler.fit_transform(relevant_parameters.reshape(-1, relevant_parameters.shape[-1])).reshape(relevant_parameters.shape)
# Save the scaler
dump(X_scaler, log_dir / 'X_scaler.joblib')
dump(Y_scaler, log_dir / 'Y_scaler.joblib')

# Parameters for k-fold validation
k = 5  # Number of folds
seed = 42  # Random seed for reproducibility
# Initialize KFold
kf = KFold(n_splits=k, shuffle=True, random_state=seed)
histories = {'mean_absolute_error':[], 'loss':[], 'val_mean_absolute_error':[], 'val_loss':[]}

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
        epochs=200, batch_size=100,
        callbacks=[tensorboard_callback],
        verbose=0
    )

    # Evaluate the model on the test set
    test_loss, test_mse, test_mae = model.evaluate(X_val, y_val)
    print(f"Test MAE: {test_mae}, Test MSE: {test_mse}")
    print(f'Score for fold {fold+1}: {model.metrics_names[0]} of {test_loss}')

    histories['mean_absolute_error'].append(history.history['mean_absolute_error'])
    histories['loss'].append(history.history['loss'])
    histories['val_mean_absolute_error'].append(history.history['val_mean_absolute_error'])
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
