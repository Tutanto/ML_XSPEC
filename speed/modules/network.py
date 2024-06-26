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