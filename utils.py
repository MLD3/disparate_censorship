import numpy as np

def one_hot(A: np.ndarray) -> np.ndarray:
    B = np.zeros((A.size, A.max() + 1))
    B[np.arange(A.size), A] = 1
    return B