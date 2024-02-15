import json
import datetime
import numpy as np

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from modules.network import r_squared, GRU_model

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

data = 'models_0.5-20_100k'

cwd = Path.cwd()
path_to_logs = cwd / 'logs'
path_to_data = cwd / 'data' / data
path_to_results = cwd / 'results'
path_to_results.mkdir(parents=True, exist_ok=True)
log_dir = path_to_logs / 'fit'
log_dir.mkdir(parents=True, exist_ok=True)
# Define the path for the model file
model_file_path = path_to_results / 'GRU_model.h5'

# File paths for the saved datasets
X_train_file = path_to_data / 'X_train_par.npy'
y_train_file = path_to_data / 'y_train_flux.npy'
X_val_file = path_to_data / 'X_val_par.npy'
y_val_file = path_to_data / 'y_val_flux.npy'
X_test_file = path_to_data / 'X_test_par.npy'
y_test_file = path_to_data / 'y_test_flux.npy'

# Check if the model file exists
if model_file_path.is_file() and X_train_file.is_file():
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
    # Load the datasets
    X = np.load(path_to_data / 'Inp_norm.npy', allow_pickle=True)
    Y = np.load(path_to_data / 'Out_norm.npy', allow_pickle=True)

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

    # Define the neural network model
    model = GRU_model(X_train_par.shape[1], y_train_flux.shape[1], neurons=128, hidden=4)
    
# Create a TensorBoard instance with log directory
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir / now, histogram_freq=1)

# Train the model
new_history = model.fit(
    X_train_par, y_train_flux,
    validation_data=(X_val_par, y_val_flux), 
    epochs=10, batch_size=16,
    callbacks=[tensorboard_callback],
    verbose=1
).history

# Save (or update) the model
model.save(model_file_path)

# Load existing history if it exists
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

# Evaluate the model on the test set
test_loss, test_mae, test_mse, test_r2 = model.evaluate(X_test_par, y_test_flux)
print(f"Test MAE: {test_mae}, Test MSE: {test_mse}, Test R2: {test_r2}")
print(f'Score: {model.metrics_names[0]} of {test_loss}')
