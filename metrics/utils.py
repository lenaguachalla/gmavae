import numpy as np

def froebenius_norm(A: np.ndarray) ->  np.ndarray:
    return np.sqrt(np.sum(A**2, axis=(-1,-2)))