from __future__ import annotations

import numpy as np


def solve_lumped_wall(
    dt: float,
    wall_T: np.ndarray,
    capacitance: np.ndarray,
    heat_source: np.ndarray,
    conductance: np.ndarray,
    boundary_T: np.ndarray,
) -> np.ndarray:
    """Implicitly advance independent lumped wall cells by one time step."""

    if dt <= 0.0:
        raise ValueError("Thermal time step must be positive")
    if np.any(capacitance <= 0.0) or np.any(conductance < 0.0):
        raise ValueError("Thermal capacitance must be positive and conductance nonnegative")
    storage = capacitance / dt
    return (storage * wall_T + heat_source + conductance * boundary_T) / (
        storage + conductance
    )
