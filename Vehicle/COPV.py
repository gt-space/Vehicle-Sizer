from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class COPV:
    volume: float
    mass: float
    diameter: float
    wall_thickness: float
    ellipse_ratio: float

    def __post_init__(self):
        if any(
            value <= 0.0
            for value in (
                self.volume,
                self.mass,
                self.diameter,
                self.wall_thickness,
                self.ellipse_ratio,
            )
        ):
            raise ValueError("COPV geometry and mass inputs must be positive")
        if self.ellipse_ratio <= 1.0:
            raise ValueError("COPV ellipse ratio must be greater than one")
        if self.inner_diameter <= 0.0:
            raise ValueError("COPV wall thickness leaves no internal diameter")
        if self.cylinder_length <= 0.0:
            raise ValueError("COPV volume is smaller than its endcap volume")

    @property
    def inner_diameter(self) -> float:
        return self.diameter - 2.0 * self.wall_thickness

    @property
    def head_depth(self) -> float:
        return 0.5 * self.inner_diameter / self.ellipse_ratio

    @property
    def cylinder_length(self) -> float:
        radius = 0.5 * self.inner_diameter
        head_volume = 4.0 / 3.0 * np.pi * radius**2 * self.head_depth
        return (self.volume - head_volume) / (np.pi * radius**2)

    @property
    def length(self) -> float:
        return self.cylinder_length + 2.0 * self.head_depth

    @property
    def internal_area(self) -> float:
        """Cylinder plus the two ellipsoidal endcaps."""

        radius = 0.5 * self.inner_diameter
        eccentricity = np.sqrt(1.0 - (self.head_depth / radius) ** 2)
        end_area = 2.0 * np.pi * radius**2 * (
            1.0
            + (1.0 - eccentricity**2)
            / eccentricity
            * np.arctanh(eccentricity)
        )
        return 2.0 * np.pi * radius * self.cylinder_length + end_area


@dataclass(frozen=True)
class COPVCatalogEntry:
    volume: float
    mass: float
    length: float
    diameter: float
    max_pressure: float

    @property
    def internal_area(self) -> float:
        radius = 0.5 * self.diameter
        sphere_volume = 4.0 / 3.0 * np.pi * radius**3
        cylinder_length = (self.volume - sphere_volume) / (np.pi * radius**2)
        return 2.0 * np.pi * radius * cylinder_length + 4.0 * np.pi * radius**2


SPECTRONIK_20L_TYPE3 = COPVCatalogEntry(0.02, 7.0, 0.661, 0.233, 350e5)
EATON_62803 = COPVCatalogEntry(0.0213, 9.1, 0.738, 0.225, 227e5)
EATON_7173 = COPVCatalogEntry(0.0226, 9.9, 0.662, 0.2642, 428e5)
EATON_62805 = COPVCatalogEntry(0.023, 9.6, 0.7783, 0.225, 227e5)
EATON_6289 = COPVCatalogEntry(0.0266, 13.6, 0.7831, 0.2436, 310e5)
EATON_6366 = COPVCatalogEntry(0.0266, 18.8, 0.7887, 0.2586, 414e5)
EATON_7130 = COPVCatalogEntry(0.0283, 12.7, 1.2195, 0.2642, 414e5)
EATON_6208 = COPVCatalogEntry(0.0313, 18.2, 1.2195, 0.2134, 345e5)
EATON_6364 = COPVCatalogEntry(0.0313, 22.0, 1.2256, 0.2184, 414e5)
EATON_7258 = COPVCatalogEntry(0.045, 11.5, 1.0668, 0.2573, 345e5)
EATON_7226 = COPVCatalogEntry(0.0452, 18.4, 1.0772, 0.2675, 379e5)
