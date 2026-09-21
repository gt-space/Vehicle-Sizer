from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Engine:
    """Structural engine properties supplied after propulsion sizing."""

    mass: float
    length: float
    exit_area: float

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in (self.mass, self.length, self.exit_area)):
            raise ValueError("Engine properties must be finite")
        if self.mass < 0.0:
            raise ValueError("Engine mass cannot be negative")
        if self.length <= 0.0 or self.exit_area <= 0.0:
            raise ValueError("Engine length and exit area must be positive")

    def axial_mass(self, resolution: int) -> np.ndarray:
        """Return the uniformly distributed engine mass."""

        if resolution < 1:
            raise ValueError("Engine mass resolution must be positive")
        return np.full(resolution, self.mass / resolution)

    def mass_on_grid(self, edges: np.ndarray, start: float) -> np.ndarray:
        """Uniform engine line density integrated over vehicle cell boundaries."""
        overlap = np.maximum(0.0, np.minimum(edges[1:], start + self.length) - np.maximum(edges[:-1], start))
        if not np.isclose(np.sum(overlap), self.length, rtol=1e-10, atol=1e-12):
            raise ValueError("Vehicle grid must contain the complete engine length")
        return self.mass * overlap / self.length
