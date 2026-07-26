import numpy as np

def vk_profile(x: np.ndarray, L: float, R: float) -> np.ndarray:
    theta = np.arccos(np.clip(1 - 2 * x / L, -1.0, 1.0))
    return (R / np.sqrt(np.pi)) * np.sqrt(theta - 0.5 * np.sin(2 * theta))

def power_series_profile(x: np.ndarray, L: float, R: float, n: float) -> np.ndarray:
    xi = np.clip(x / L, 0.0, 1.0)
    return R * xi**n

def annulus_volume(r_o, r_i, L):
    return np.pi * (r_o**2 - r_i**2) * L

def annulus_second_moment(r_o, r_i):
    return 0.25 * np.pi * (r_o**4 - r_i**4)

def cylinder_lateral_area(r, L):
    return 2 * r * L

def cylinder_surface_area(r, L):
    return 2 * np.pi * r * L
