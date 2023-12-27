import json
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
scaler = load(log_dir / 'scaler.joblib')

# Get a list of all JSONS files in the specified directory
json_files = list(path_to_models.glob("model_*.json"))


for i, file_path in enumerate(json_files):
    # Read the JSON data from the file
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # Extract energy and flux from the JSON data (replace 'Energy' and 'Flux' with your actual keys)
    flux = json_data['flux (1 / keV cm^-2 s)']
    parameters = json_data['parameters']
    relevant_parameters, removed_columns = remove_uniform_columns(parameters)

    X_new_scaled = scaler.transform(flux)

    # Make predictions
    predictions = model.predict(X_new_scaled)

    predictions = scaler.inverse_transform(predictions)

