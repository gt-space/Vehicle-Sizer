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
