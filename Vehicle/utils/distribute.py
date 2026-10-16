import numpy as np

def uniform(total: float, n: int) -> np.ndarray:
    return np.full(n, total / n)
