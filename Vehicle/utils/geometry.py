import numpy as np

def vk_profile(x: np.ndarray, L: float, R: float) -> np.ndarray:
    theta = np.arccos(np.clip(1 - 2 * x / L, -1.0, 1.0))
    return (R / np.sqrt(np.pi)) * np.sqrt(theta - 0.5 * np.sin(2 * theta))

def power_series_profile(x: np.ndarray, L: float, R: float, n: float) -> np.ndarray:
    xi = np.clip(x / L, 0.0, 1.0)
    return R * xi**n

def annulus_volume(r_o, r_i, L):
    return np.pi * (r_o**2 - r_i**2) * L

def pressure_wall_thickness(pressure, diameter, fos, allowable):
    """Pressure-sized tank gauge using the configured allowable stress."""
    if (not np.isfinite([pressure, diameter, fos, allowable]).all()
            or pressure <= 0 or diameter <= 0 or fos < 1 or allowable <= 0):
        raise ValueError("Tank pressure, diameter and allowable must be positive; pressure FOS >= 1")
    return fos * pressure * diameter / (2 * allowable)

def tank_shell_volume(diameter, thickness, cylinder_length, ellipse_ratio,
                      endcap_multiplier, passthrough_diameter=0., passthrough_thickness=0.):
    """Metal shell volume: cylinder, ellipsoidal heads and optional passthrough."""
    if not np.isfinite(endcap_multiplier) or endcap_multiplier <= 0:
        raise ValueError("endcap_mass_multiplier must be finite and positive")
    a = diameter / 2
    c = a / ellipse_ratio
    area = 4 * np.pi * ((a**3.2 + 2 * (a*c)**1.6) / 3)**(1/1.6)
    volume = area * thickness * endcap_multiplier
    volume += annulus_volume(a, a - thickness, cylinder_length)
    if passthrough_diameter > 0:
        if passthrough_thickness >= passthrough_diameter / 2:
            raise ValueError("Passthrough wall consumes its internal diameter")
        volume += annulus_volume(passthrough_diameter / 2,
                                 passthrough_diameter / 2 - passthrough_thickness,
                                 cylinder_length)
    return volume

def annulus_second_moment(r_o, r_i):
    return 0.25 * np.pi * (r_o**4 - r_i**4)

def cylinder_lateral_area(r, L):
    return 2 * r * L

def cylinder_surface_area(r, L):
    return 2 * np.pi * r * L
