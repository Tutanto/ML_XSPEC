def Model(x, par):
    x_grid = (np.array(x).reshape(-1, n_points).mean(axis=1)).tolist()
    y_pred = model.predict(par.reshape(-1, par.shape[0]), verbose=0)
    y_pred_d = flux_scaler.inverse_transform(y_pred)

    return np.interp(x, x_grid, y_pred_d[0])

def log_likelihood(theta):
    model_par = theta[:-1].copy()
    log_f = theta[-1]
    model = Model(xd, model_par)
    sigma2 = yderr**2 + model**2 * np.exp(2 * log_f)
#    sigma2 = yerr**2 + np.exp(2 * log_f)
    return -0.5 * np.sum((yd - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior(params):
    # Assuming params is an instance of Parameters class

    # Automatically check each parameter in the params object
    for param_name in params.__dict__:
        # Skip checking for attributes that are not parameters (like 'ranges')
        if param_name == 'ranges':
            continue

        if not params.is_param_within_range(param_name):
            return -np.inf

    # If all parameters are within range, return 0.0 or another suitable value
    return 0.0

def log_probability(theta):
    params = Parameters()
    params.update_from_array(theta)
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf, None
    return lp + log_likelihood(theta), lp