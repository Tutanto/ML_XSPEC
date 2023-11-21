import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from modules.utils import process_ipac_files

path_to_models = Path(Path.cwd() / 'models')
# Assuming 'df' is your DataFrame with Flux distributions and parameters
df = process_ipac_files(path_to_models)

# Extract Flux distributions and parameters
X_flux = df['Flux'].tolist()
y_params = df[['LineE', 'Betor10', 'Rin_M', 'Rout_M', 'Incl', 'norm']].values

# Convert Flux distributions to NumPy arrays
X_flux_array = np.array(X_flux)

# Split the data into training, validation, and test sets
X_train_flux, X_temp_flux, y_train, y_temp = train_test_split(X_flux_array, y_params, test_size=0.3, random_state=42)
X_val_flux, X_test_flux, y_val, y_test = train_test_split(X_temp_flux, y_temp, test_size=0.5, random_state=42)

# Standardize the Flux distributions
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flux)
X_val_scaled = scaler.transform(X_val_flux)
X_test_scaled = scaler.transform(X_test_flux)

# Define the neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_params.shape[1]))  # Output layer with the number of parameters as neurons

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=50, batch_size=32)

# Evaluate the model on the test set
mse = model.evaluate(X_test_scaled, y_test)
print(f'Mean Squared Error on Test Set: {mse}')

# Make predictions on new data (assuming 'X_new_flux' is the Flux distribution of a new example)
X_new_scaled = scaler.transform(X_new_flux.reshape(1, -1))
predictions = model.predict(X_new_scaled)
print('Predicted Parameters:', predictions)
