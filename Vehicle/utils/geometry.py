import numpy as np

def nose_profile(x: np.ndarray, length: float, radius: float, definition: dict) -> np.ndarray:
    """Use the same configured nose contour for both nosecone and avionics bay."""
    profile = definition["profile"]
    if profile == "von_karman":
        theta = np.arccos(np.clip(1 - 2 * x / length, -1.0, 1.0))
        return (radius / np.sqrt(np.pi)) * np.sqrt(theta - 0.5 * np.sin(2 * theta))
    if profile == "power_series":
        return radius * np.clip(x / length, 0.0, 1.0)**definition.get("power_series_n", 0.66)
    raise ValueError(f"Unknown nosecone profile: {profile}")

def annulus_volume(r_o, r_i, L):
    return np.pi * (r_o**2 - r_i**2) * L

def pressure_wall_thickness(pressure, diameter, fos, allowable):
    """Pressure-sized tank gauge using the configured allowable stress."""
    if (not np.isfinite([pressure, diameter, fos, allowable]).all()
            or pressure <= 0 or diameter <= 0 or fos < 1 or allowable <= 0):
        raise ValueError("Tank pressure, diameter and allowable must be positive; pressure FOS >= 1")
    return fos * pressure * diameter / (2 * allowable)

def tank_capacity(radius, passthrough_radius, ellipse_ratio):
    """Usable volume of both heads and cylinder cross-section [m³, m²]."""
    if not 0 <= passthrough_radius < radius or ellipse_ratio <= 1:
        raise ValueError("Tank requires positive bore and ellipse ratio greater than one")
    head_depth = radius / ellipse_ratio
    beta = np.sqrt(1 - (passthrough_radius / radius)**2)
    return (4 / 3 * np.pi * radius**2 * head_depth * beta**3,
            np.pi * (radius**2 - passthrough_radius**2))

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
