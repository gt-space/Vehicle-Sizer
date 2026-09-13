from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Engine:
    """Structural engine properties supplied after propulsion sizing."""

    mass: float
    length: float
    exit_area: float

    def __post_init__(self) -> None:
        if self.mass < 0.0:
            raise ValueError("Engine mass cannot be negative")
        if self.length <= 0.0 or self.exit_area <= 0.0:
            raise ValueError("Engine length and exit area must be positive")

    def axial_mass(self, resolution: int) -> np.ndarray:
        """Return the uniformly distributed engine mass."""

        if resolution < 1:
            raise ValueError("Engine mass resolution must be positive")
        return np.full(resolution, self.mass / resolution)
