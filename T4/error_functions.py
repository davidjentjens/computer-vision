import numpy as np

def mse(y, y_pred):
    return np.mean(np.power(y - y_pred, 2))

def mse_prime(y, y_pred):
    return 2 * (y_pred - y) / y.size
