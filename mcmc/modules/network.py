import tensorflow as tf
from tensorflow.keras import backend as K


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