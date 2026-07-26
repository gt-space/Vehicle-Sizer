import numpy as np

def nosecone_CNa(M: float) -> float:
    return 2 * get_comp_factor(M)

def body_CNa(M: float, alpha: float, A_plan: float, A_ref: float) -> float:
    return 1.1 * get_comp_factor(M) * (A_plan / A_ref) * (np.sin(alpha)**2 / alpha)

def taper_CNa(M: float, alpha: float, A_exit: float, A_ref: float) -> float:
    return get_comp_factor(M) * (2 / A_ref) * (A_exit - A_ref) * (np.sin(alpha) / alpha)

def fins_CNa(M: float, N: int, s: float, Cr: float, Ct: float, R_ref: float) -> float:
    Kfb = 1 + (R_ref / (s + R_ref))
    CNa1 = (4 * N * (s / (R_ref * 2))**2) / (1 + np.sqrt(1 + (s / (Cr + Ct))**2))
    return get_comp_factor(M) * (N * 0.5) * Kfb * CNa1

def get_comp_factor(M: float) -> float:
    if M <= 0.8:
        return 1 / np.sqrt(1 - M**2)
    elif M < np.sqrt(34 / 5):
        return 1 / np.sqrt(1 - 0.8**2)
    else:
        return 1 / np.sqrt(M**2 - 1)