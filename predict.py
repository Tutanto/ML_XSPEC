import json
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from pathlib import Path
from tensorflow.keras.models import load_model

from modules.utils import remove_uniform_columns

path_to_models = Path(Path.cwd() / 'all_models' / 'test_models')
path_to_logs = Path(Path.cwd() / 'logs')
log_dir = path_to_logs / 'fit'

# Load the model
model = load_model(log_dir /'my_model.h5')
# Load the saved scaler
X_scaler = load(log_dir / 'X_scaler.joblib')
Y_scaler = load(log_dir / 'Y_scaler.joblib')

# Get a list of all JSONS files in the specified directory
json_files = list(path_to_models.glob("model_*.json"))

fluxes = []
parameters = []
predictions = []

for i, file_path in enumerate(json_files):
    # Read the JSON data from the file
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # Extract energy and flux from the JSON data (replace 'Energy' and 'Flux' with your actual keys)
    fluxes.append(np.array(json_data['flux (1 / keV cm^-2 s)']))
    parameters.append(list(json_data['parameters'].values()))

relevant_parameters, removed_columns = remove_uniform_columns(np.array(parameters))

for i,flux in enumerate(fluxes):
    X_new_scaled = X_scaler.transform(flux.reshape(-1, flux.shape[-1])).reshape(flux.shape)

    # Make predictions
    prediction = model.predict(X_new_scaled.reshape(1, X_new_scaled.shape[0]))
    predictions.append(Y_scaler.inverse_transform(prediction))

# Ensure that each model's predictions are 2D (num_samples x num_features)
predictions_array = [pred.squeeze() for pred in predictions]
predictions_array = [pred.reshape(1, -1) if pred.ndim == 1 else pred for pred in predictions_array]

# Convert to numpy array for easier manipulation
predictions_array = np.array(predictions_array)

# Diagnostic print statements
print("Shapes of predictions from each model:")
for model_idx, pred in enumerate(predictions_array):
    print(f"Model {model_idx + 1}: {pred.shape}")

# Number of parameters to plot
num_parameters = relevant_parameters.shape[1]

# Plotting
plt.figure(figsize=(15, num_parameters * 5))

for i in range(num_parameters):
    plt.subplot(num_parameters, 1, i + 1)
    plt.plot(predictions_array[:, 0, i], label=f'Model {model_idx + 1}', linestyle='-')
    plt.plot(relevant_parameters[:, i], label='True Parameter', linestyle='--')
    plt.title(f'Parameter {i + 1} Comparison')
    plt.legend()

plt.tight_layout()
plt.show()