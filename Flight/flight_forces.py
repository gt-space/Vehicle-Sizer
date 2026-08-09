import numpy as np

def drag(Cd:float, q: float, A_ref: float) -> float:
    return Cd * q * A_ref

def gravity(m: float, h: float) -> float:
    g0 = 9.80665
    Re = 6378137
    return m * g0 * (Re / (Re + h))**2