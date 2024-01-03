import os
import h5py
import json
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential, load_model


from modules.utils import process_json_files_batch, combine_hdf5_files, remove_uniform_columns


def adjusted_r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    r2 = 1 - SS_res/(SS_tot + K.epsilon())
    
    n = tf.cast(tf.shape(y_true)[0], tf.float32)  # Number of data points
    p = tf.cast(tf.shape(y_true)[1], tf.float32)  # Number of predictors/features

    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

path_to_models = Path(Path.cwd() / 'models_0.1-100')
path_to_logs = Path(Path.cwd() / 'logs')
path_to_batches = Path(Path.cwd() / 'batches')
path_to_batches.mkdir(parents=True, exist_ok=True)
log_dir = path_to_logs / 'fit'
log_dir.mkdir(parents=True, exist_ok=True)
# Define the path for the model file
model_file_path = log_dir / 'my_model.h5'

# File paths for the saved datasets
X_train_file = log_dir / 'X_train_flux.npy'
y_train_file = log_dir / 'y_train.npy'
X_val_file = log_dir / 'X_val_flux.npy'
y_val_file = log_dir / 'y_val.npy'
X_test_file = log_dir / 'X_test_flux.npy'
y_test_file = log_dir / 'y_test.npy'

# Check if the model file exists
if model_file_path.is_file() and X_train_file.is_file() and X_train_file.is_file():
    # Load the datasets
    print("Loading the datasets...")
    # Load the datasets
    X_train_flux = np.load(X_train_file, allow_pickle=True)
    y_train = np.load(y_train_file, allow_pickle=True)
    X_val_flux = np.load(X_val_file, allow_pickle=True)
    y_val = np.load(y_val_file, allow_pickle=True)
    X_test_flux = np.load(X_test_file, allow_pickle=True)
    y_test = np.load(y_test_file, allow_pickle=True)
    print("Loading the saved model...")
    model = load_model(model_file_path, custom_objects={'adjusted_r_squared': adjusted_r_squared})
else:
    print("No saved model found. Using a new model...")
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
    scaler = StandardScaler()
    all_flux_values_normalized = scaler.fit_transform(all_flux_values.reshape(-1, all_flux_values.shape[-1])).reshape(all_flux_values.shape)
    relevant_parameters_normalized = scaler.fit_transform(relevant_parameters.reshape(-1, relevant_parameters.shape[-1])).reshape(relevant_parameters.shape)

    # Split the data into training, validation, and test sets
    X_train_flux, X_temp_flux, y_train, y_temp = train_test_split(
        all_flux_values_normalized, 
        relevant_parameters_normalized, 
        test_size=0.3, random_state=42
        )
    X_val_flux, X_test_flux, y_val, y_test = train_test_split(
        X_temp_flux, 
        y_temp, 
        test_size=0.5, random_state=42
        )

    # Save the datasets
    np.save(X_train_file, X_train_flux)
    np.save(y_train_file, y_train)
    np.save(X_val_file, X_val_flux)
    np.save(y_val_file, y_val)
    np.save(X_test_file, X_test_flux)
    np.save(y_test_file, y_test)

    # Define the neural network model
    model = Sequential()
    model.add(GRU(units=128, return_sequences=True, input_shape=(X_train_flux.shape[1], 1)))
    model.add(GRU(units=128, return_sequences=True))
    model.add(GRU(units=128, return_sequences=True))
    model.add(GRU(units=128, return_sequences=True))
    model.add(GRU(units=128, return_sequences=False))
    model.add(Dense(y_train.shape[1], activation='linear'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.00001, clipnorm=1.0), 
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_logarithmic_error',
                'mean_absolute_percentage_error', adjusted_r_squared] # List of metrics
        ) 
    
# Create a TensorBoard instance with log directory
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir / now, histogram_freq=1)

# Train the model
new_history = model.fit(
    X_train_flux, y_train,
    validation_data=(X_val_flux, y_val), 
    epochs=25, batch_size=50,
    callbacks=[tensorboard_callback],
    verbose=1
).history

# Save (or update) the model
model.save(model_file_path)

# Load existing history if it exists
history_filename = log_dir / 'training_history.json'
if os.path.exists(history_filename):
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

# Evaluate the model on the test set
test_loss, test_mae, test_msle, test_mape, test_adj = model.evaluate(X_test_flux, y_test)
print(f"Test MAE: {test_mae}, Test MSLE: {test_msle}")
print(f'Score: {model.metrics_names[0]} of {test_loss}')

history_df = pd.DataFrame(existing_history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['mean_absolute_error', 'val_mean_absolute_error']].plot()
history_df.loc[:, ['mean_squared_logarithmic_error', 'val_mean_squared_logarithmic_error']].plot()
history_df.loc[:, ['mean_absolute_percentage_error', 'val_mean_absolute_percentage_error']].plot()
history_df.loc[:, ['adjusted_r_squared', 'val_adjusted_r_squared']].plot()
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
