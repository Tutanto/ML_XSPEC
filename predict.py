import json
import numpy as np
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
    prediction = model.predict(X_new_scaled)
    predictions.append(Y_scaler.inverse_transform(prediction))
