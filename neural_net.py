import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

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

# Normalize the Flux distributions
scaler_x = MinMaxScaler()
X_train_scaled = scaler_x.fit_transform(X_train_flux)
X_val_scaled = scaler_x.transform(X_val_flux)
X_test_scaled = scaler_x.transform(X_test_flux)

# Normalize the target variables using MinMaxScaler
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# Define the neural network model
model = Sequential()
model.add(Dense(1024, input_dim=X_train_scaled.shape[1], activation='relu'))
#model.add(Dropout(0.3))
#model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='linear'))  # Output layer with the number of parameters as neurons

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.00001, clipvalue=0.5), loss='mse', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_scaled, y_train_scaled, 
    validation_data=(X_val_scaled, y_val_scaled), 
    epochs=100, batch_size=150)

# Evaluate the model on the test set
mse = model.evaluate(X_test_scaled, y_test_scaled)
print(f'Mean Squared Error on Test Set: {mse}')

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

# Make predictions on new data (assuming 'X_new_flux' is the Flux distribution of a new example)
#X_new_scaled = scaler.transform(X_new_flux.reshape(1, -1))
#predictions = model.predict(X_new_scaled)
#print('Predicted Parameters:', predictions)
