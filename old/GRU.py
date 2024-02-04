import h5py
import json
import datetime
import numpy as np
from joblib import dump
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model


from modules.utils import process_json_files_batch, combine_hdf5_files, remove_uniform_columns
from modules.network import r_squared, GRU_model

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

path_to_models = Path(Path.cwd() / 'all_models' / 'models_0.5-20_10k')
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
    model = load_model(model_file_path, custom_objects={'r_squared': r_squared})
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
    X_scaler = MinMaxScaler()
    all_flux_values_normalized = X_scaler.fit_transform(all_flux_values.reshape(-1, all_flux_values.shape[-1])).reshape(all_flux_values.shape)
    # Save the scaler
    dump(X_scaler, log_dir / 'X_scaler.joblib')
    # Find the indices of rows that contain NaN
    indices_with_nan = np.any(np.isnan(all_flux_values_normalized), axis=1)
    # Count the rows to be removed
    rows_to_remove = np.sum(indices_with_nan)
    # Remove the rows that contain NaN
    all_flux_values_normalized = all_flux_values_normalized[~indices_with_nan]
    Y = Y[~indices_with_nan]
    # Print the number of rows removed
    print(f"Number of rows removed: {rows_to_remove}")

    # Split the data into training, validation, and test sets
    X_train_flux, X_temp_flux, y_train, y_temp = train_test_split(
        all_flux_values_normalized, 
        Y, 
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
    model = GRU_model(X_train_flux.shape[1], y_train.shape[1])
    
# Create a TensorBoard instance with log directory
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir / now, histogram_freq=1)

# Train the model
new_history = model.fit(
    X_train_flux, y_train,
    validation_data=(X_val_flux, y_val), 
    epochs=10, batch_size=16,
    callbacks=[tensorboard_callback],
    verbose=1
).history

# Save (or update) the model
model.save(model_file_path)

# Load existing history if it exists
history_filename = log_dir / 'training_history.json'
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

# Evaluate the model on the test set
test_loss, test_mae, test_mse, test_r2 = model.evaluate(X_test_flux, y_test)
print(f"Test MAE: {test_mae}, Test MSE: {test_mse}, Test R2: {test_r2}")
print(f'Score: {model.metrics_names[0]} of {test_loss}')
