import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam

def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 

    return 1 - SS_res/(SS_tot + K.epsilon())

def adjusted_r_squared(y_true, y_pred):
    r2 = r_squared(y_true, y_pred)
    
    n = tf.cast(tf.shape(y_true)[0], tf.float32)  # Number of data points
    p = tf.cast(tf.shape(y_true)[1], tf.float32)  # Number of predictors/features

    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

def ANN_model(input_dim, output_dim):
    # Define the neural network model
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(512, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(512, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(512, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(512, activation='relu', kernel_initializer=HeNormal()))
    model.add(Dense(output_dim, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.000001, clipnorm=1.0), 
        loss='mean_squared_logarithmic_error', 
        metrics=['mean_squared_error',
                 'mean_absolute_error',
                 r_squared] # List of metrics
        )
    
    return model

def GRU_model(input_dim, output_dim):
    # Define the neural network model
    model = Sequential()
    model.add(GRU(units=256, return_sequences=True, input_shape=(input_dim, 1)))
    model.add(GRU(units=256, return_sequences=True))
    model.add(GRU(units=256, return_sequences=True))
    model.add(GRU(units=256, return_sequences=True))
    model.add(GRU(units=256, return_sequences=False))
    model.add(Dense(output_dim, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.000001, clipnorm=1.0), 
        loss='mean_squared_logarithmic_error',
        metrics=['mean_absolute_error', 
                 'mean_squared_error',
                 r_squared] # List of metrics
        )
    
    return model

# Function to calculate mean and standard deviation across folds for each epoch
def calc_mean_std_per_epoch(data):
    # Transpose to make each row represent an epoch and each column a fold
    transposed_data = list(map(list, zip(*data)))
    means = [np.mean(epoch_data) for epoch_data in transposed_data]
    stds = [np.std(epoch_data) for epoch_data in transposed_data]
    return means, stds

# Plotting function for two metrics
def plot_two_metrics(epochs, mean_values1, std_values1, label1, mean_values2, std_values2, label2, title):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, mean_values1, label=f'Mean {label1}')
    plt.fill_between(epochs, np.array(mean_values1) - np.array(std_values1), 
                     np.array(mean_values1) + np.array(std_values1), alpha=0.2)
    plt.plot(epochs, mean_values2, label=f'Mean {label2}', linestyle='--')
    plt.fill_between(epochs, np.array(mean_values2) - np.array(std_values2), 
                     np.array(mean_values2) + np.array(std_values2), alpha=0.2, linestyle='--')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.show()