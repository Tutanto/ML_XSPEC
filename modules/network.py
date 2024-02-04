import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
    GRU, 
    Dense, 
    Dropout, 
    Activation, 
    BatchNormalization
)


def r_squared(y_true, y_pred):
    """
    Calculates the coefficient of determination, R^2, for the prediction.

    This function computes R^2, a statistical measure of how well the 
    regression predictions approximate the real data points. An R^2 of
    1 indicates perfect correlation.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted labels.

    Returns:
    tensor: The R^2 value.
    """
    SS_res = K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 

    return 1 - SS_res/(SS_tot + K.epsilon())

def adjusted_r_squared(y_true, y_pred):
    """
    Calculates the adjusted R^2, accounting for the number of predictors in the model.

    This function modifies the R^2 value to penalize the model complexity 
    (number of predictors). It's useful for comparing models with a different 
    number of predictors.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted labels.

    Returns:
    tensor: The adjusted R^2 value.
    """
    r2 = r_squared(y_true, y_pred)
    
    n = tf.cast(tf.shape(y_true)[0], tf.float32)  # Number of data points
    p = tf.cast(tf.shape(y_true)[1], tf.float32)  # Number of predictors/features

    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


def ANN_model(input_dim, output_dim, neurons=128, hidden=4, dropout=.3, learning_rate=1.e-4):
    """
    Constructs an Artificial Neural Network (ANN) model using Keras.

    The model consists of one input layer, multiple hidden layers, and one output layer.
    Each hidden layer is followed by ReLU activation, Dropout, and BatchNormalization.
    The model uses the Adam optimizer and is compiled with specific loss and metrics.

    Parameters:
    - input_dim (int): The number of input features. This is the size of the first dimension of the input data.
    - output_dim (int): The number of neurons in the output layer. Determines the size of the output dimension.
    - neurons (int, optional): The number of neurons in each hidden layer. Defaults to 128.
    - hidden (int, optional): The number of hidden layers in the network. Defaults to 4.
    - dropout (float, optional): The dropout rate for regularization, applied to each hidden layer. Defaults to 0.3.
    - learning_rate (float, optional): The learning rate for the Adam optimizer. Defaults to 1.e-4.

    Returns:
    - model: A Keras Sequential model, compiled and ready for training.

    Example:
    ```
    model = ANN_model(input_dim=100, output_dim=1)
    ```

    This function creates an ANN model with 100 input features, 1 output neuron, 4 hidden layers (each with 128 neurons),
    a dropout rate of 0.3, and a learning rate of 0.0001 for the optimizer.
    """
    # Define the neural network model
    model = Sequential()
    
    # Input layer with BatchNormalization
    model.add(Dense(neurons, input_dim=input_dim, kernel_initializer=HeNormal()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # Hidden layers with Dropout and BatchNormalization
    for _ in range(hidden):
        model.add(Dense(neurons, kernel_initializer=HeNormal()))
        model.add(Activation('relu'))
        if dropout:
            model.add(Dropout(dropout)) # Apply Dropout, adjust the dropout rate as needed
        model.add(BatchNormalization()) # Apply BatchNormalization

    # Output layer
    model.add(Dense(output_dim, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), 
        loss='mean_squared_logarithmic_error', 
        metrics=['mean_squared_error', 'mean_absolute_error', r_squared]
    )
    
    return model


def GRU_model(input_dim, output_dim, neurons=128, hidden=2, learning_rate=1.e-4):
    """
    Constructs a Gated Recurrent Unit (GRU) based neural network model using Keras.

    This model is designed for sequence prediction tasks and includes one input GRU layer,
    multiple hidden GRU layers, and one output Dense layer.

    Parameters:
    - input_dim (int): The number of time steps (sequence length) in each input sample.
    - output_dim (int): The number of neurons in the output layer. Determines the size of the model's output.
    - neurons (int, optional): The number of units (neurons) in each GRU layer. Defaults to 128.
    - hidden (int, optional): The number of hidden GRU layers in the network. Defaults to 4.
    - learning_rate (float, optional): The learning rate for the Adam optimizer. Defaults to 1.e-4.

    Returns:
    - model: A Keras Sequential model, compiled and ready for training.

    Example:
    ```
    model = GRU_model(input_dim=10, output_dim=1)
    ```

    This function creates a GRU-based model with an input sequence length of 10, 1 output neuron,
    4 hidden layers (each with 128 neurons), a dropout rate of 0.3 for both inputs and recurrent connections,
    and a learning rate of 0.0001 for the optimizer.
    """
    # Define the neural network model
    model = Sequential()
    # Input layer
    model.add(GRU(units=neurons, return_sequences=True, input_shape=(input_dim, 1)))
    # Hidden layers with Dropout
    for _ in range(hidden):
        model.add(GRU(units=neurons, return_sequences=True))
    model.add(GRU(units=neurons, return_sequences=False))
    # Output layer
    model.add(Dense(output_dim, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), 
        loss='mean_squared_logarithmic_error',
        metrics=['mean_absolute_error', 
                 'mean_squared_error',
                 r_squared] # List of metrics
        )
    
    return model

def calc_mean_std_per_epoch(data):
    """
    Calculates mean and standard deviation for each epoch across different folds.

    Useful in scenarios like k-fold cross-validation where you want to 
    track the performance per epoch across different folds.

    Parameters:
    data (list of lists): Nested list where each inner list contains 
                          metrics of each fold for a particular epoch.

    Returns:
    list, list: Lists containing mean and standard deviation for each epoch.
    """
    # Transpose to make each row represent an epoch and each column a fold
    transposed_data = list(map(list, zip(*data)))
    means = [np.mean(epoch_data) for epoch_data in transposed_data]
    stds = [np.std(epoch_data) for epoch_data in transposed_data]
    return means, stds

def plot_two_metrics(epochs, mean_values1, std_values1, label1, mean_values2, std_values2, label2, title):
    """
    Plots two metrics over epochs with their means and standard deviations.

    Parameters:
    epochs (list or array): List or array of epoch numbers.
    mean_values1 (list or array): Mean values of the first metric for each epoch.
    std_values1 (list or array): Standard deviation values of the first metric for each epoch.
    label1 (str): Label for the first metric.
    mean_values2 (list or array): Mean values of the second metric for each epoch.
    std_values2 (list or array): Standard deviation values of the second metric for each epoch.
    label2 (str): Label for the second metric.
    title (str): Title of the plot.

    Returns:
    None: This function does not return anything but plots the metrics.
    """
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