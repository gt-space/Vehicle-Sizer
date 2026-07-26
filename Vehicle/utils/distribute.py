import numpy as np

def uniform(total: float, n: int) -> np.ndarray:
    return np.full(n, total / n)

def uniform_full(total: float, n: int) -> np.ndarray:
    return np.full(n, total)

def weighted(total: float, weights: np.ndarray) -> np.ndarray:
    return total * weights / np.sum(weights)