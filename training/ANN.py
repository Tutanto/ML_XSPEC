import json
import datetime
import numpy as np

from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import TensorBoard

from modules.network import ANN_model

from modules.variables import (
    path_to_logs,
    path_to_data,
    path_to_results
)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

data = 'models_0.5-20_100k_smooth_5'

path_to_data = path_to_data / data
path_to_results.mkdir(parents=True, exist_ok=True)
log_dir = path_to_logs / 'fit'
log_dir.mkdir(parents=True, exist_ok=True)

# Load the datasets
X = np.load(path_to_data / 'Inp.npy', allow_pickle=True)
Y = np.load(path_to_data / 'Out_norm.npy', allow_pickle=True)

# Parameters for k-fold validation
k = 5  # Number of folds
seed = 42  # Random seed for reproducibility
# Initialize KFold
kf = KFold(n_splits=k, shuffle=True, random_state=seed)
histories = {'mean_absolute_error':[], 'mean_squared_error':[], 'r_squared':[],  
             'loss':[], 'val_mean_absolute_error':[], 'val_mean_squared_error': [], 
             'val_r_squared':[], 'val_loss':[]}

# Loop over each fold
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Training on fold {fold+1}/{k}...")

    # Splitting data into training and validation sets for this fold
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = Y[train_index], Y[val_index]
        
    # Create a TensorBoard instance with log directory
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir / now, histogram_freq=1)

    model = ANN_model(X_train.shape[1], y_train.shape[1], neurons=256, hidden=6, dropout=0.3)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val), 
        epochs=250, batch_size=50,
        callbacks=[tensorboard_callback],
        verbose=0
    )

    # Evaluate the model on the test set
    test_loss, test_mse, test_mae, test_r2 = model.evaluate(X_val, y_val)
    print(f"Test MAE: {test_mae}, Test MSE: {test_mse}, Test R2: {test_r2}")
    print(f'Score for fold {fold+1}: {model.metrics_names[0]} of {test_loss}')

    histories['mean_absolute_error'].append(history.history['mean_absolute_error'])
    histories['mean_squared_error'].append(history.history['mean_squared_error'])
    histories['r_squared'].append(history.history['r_squared'])
    histories['loss'].append(history.history['loss'])
    histories['val_mean_absolute_error'].append(history.history['val_mean_absolute_error'])
    histories['val_mean_squared_error'].append(history.history['val_mean_squared_error'])
    histories['val_r_squared'].append(history.history['val_r_squared'])    
    histories['val_loss'].append(history.history['val_loss'])

model.save(path_to_results / 'ANN_model.h5')

# Save merged history
history_filename = path_to_results / 'ANN_training_history.json'
with open(history_filename, 'w') as f:
    json.dump(histories, f)
