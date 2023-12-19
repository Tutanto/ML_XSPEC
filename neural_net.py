import h5py
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from modules.utils import process_json_files_batch, combine_hdf5_files

path_to_models = Path(Path.cwd() / 'models')
path_to_logs = Path(Path.cwd() / 'logs')
path_to_batches = Path(Path.cwd() / 'batches')
path_to_batches.mkdir(parents=True, exist_ok=True)
log_dir = path_to_logs / 'fit'
log_dir.mkdir(parents=True, exist_ok=True)

for i, (flux, params) in enumerate(process_json_files_batch(path_to_models)):
    with h5py.File(path_to_batches / f'batch_{i}.h5', 'w') as hf:
        hf.create_dataset('flux', data=flux)
        hf.create_dataset('params', data=params)

# Example usage
all_flux_values, all_parameters = combine_hdf5_files(path_to_batches)
# Convert y_train to a float data type
all_parameters = np.array(all_parameters, dtype=float)

# Normalize flux values
scaler = StandardScaler()
all_flux_values_normalized = scaler.fit_transform(all_flux_values.reshape(-1, all_flux_values.shape[-1])).reshape(all_flux_values.shape)

# Split the data into training, validation, and test sets
X_train_flux, X_temp_flux, y_train, y_temp = train_test_split(all_flux_values_normalized, all_parameters, test_size=0.3, random_state=42)
X_val_flux, X_test_flux, y_val, y_test = train_test_split(X_temp_flux, y_temp, test_size=0.5, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train_flux.shape[1], activation='relu', kernel_initializer=HeNormal()))
#model.add(Dropout(0.5))
#model.add(BatchNormalization())
'''model.add(Dense(512, activation='relu', kernel_initializer=HeNormal()))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu', kernel_initializer=HeNormal()))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu', kernel_initializer=HeNormal()))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu', kernel_initializer=HeNormal()))
model.add(Dropout(0.5))
model.add(BatchNormalization())'''
model.add(Dense(8, activation='relu', kernel_initializer=HeNormal()))
#model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Dense(y_train.shape[1], activation='linear'))  # Output layer with the number of parameters as neurons

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mean_squared_logarithmic_error', metrics=['accuracy'])
#model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='mse', metrics=['accuracy'])
#model.compile(optimizer=RMSprop(learning_rate=0.001), loss='mean_absolute_error', metrics=['accuracy'])

# Create a TensorBoard instance with log directory
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir / now, histogram_freq=1)

# Train the model
history = model.fit(
    X_train_flux, y_train, 
    validation_data=(X_val_flux, y_val), 
    epochs=100, batch_size=150,
    callbacks=[tensorboard_callback])

# Evaluate the model on the test set
mse = model.evaluate(X_test_flux, y_test)
print(f'Mean Squared Error on Test Set: {mse}')

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

# Make predictions on new data (assuming 'X_new_flux' is the Flux distribution of a new example)
#X_new_scaled = scaler.transform(X_new_flux.reshape(1, -1))
#predictions = model.predict(X_new_scaled)
#print('Predicted Parameters:', predictions)
