"""Quadratic pressure-rise curves: kg/s in, Pa out (not head in metres).

Reference shape comes from [mdot, dP] points or polynomial coefficients [a,b,c].
Scale horizontally by design/reference flow and vertically by design pressure
divided by the fitted reference pressure, so the design point is exact.
"""
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np


@dataclass(frozen=True)
class PumpCurve:
    a: float
    b: float
    c: float
    max_mdot: float

    def __call__(self, mdot: float) -> float:
        # Solver trials may leave the operating domain. Validate accepted states
        # separately rather than clipping the polynomial and hiding violations.
        value = (self.a * mdot + self.b) * mdot + self.c
        if not isfinite(value):
            raise ValueError("Nonfinite pump curve evaluation")
        return value

    def validate_flow(self, mdot: float) -> None:
        tolerance = 1e-10 * max(self.max_mdot, 1.0)
        if not isfinite(mdot) or not -tolerance <= mdot <= self.max_mdot + tolerance:
            raise ValueError(f"Pump curve flow {mdot} kg/s outside [0, {self.max_mdot}]")


def scaled_pump_curve(mdot: float, dP: float, *, reference_mdot: float,
                      points=None, coefficients=None) -> PumpCurve:
    """Fit once per design; return a cheap callable used by the fluid solver.

    Coefficients are descending powers of reference mass flow. This model
    supports decreasing, concave-down quadratics only. Its operating domain
    ends at zero pressure rise, or the highest supplied flow, whichever is less.
    It does not model reverse operation, speed control, efficiency or cavitation.
    """
    if any(not isfinite(x) or x <= 0 for x in (mdot, dP, reference_mdot)):
        raise ValueError("Pump design flow, pressure rise and reference flow must be positive")
    if (points is None) == (coefficients is None):
        raise ValueError("Supply either reference points or coefficients, not both")
    maximum = float("inf")
    if points is not None:
        data = np.asarray(points, dtype=float)
        if (data.ndim != 2 or data.shape[1] != 2 or len(data) < 3
                or not np.all(np.isfinite(data)) or np.any(data < 0)
                or len(np.unique(data[:, 0])) != len(data)):
            raise ValueError("Pump reference requires >=3 finite nonnegative [mdot, dP] points with distinct flows")
        if data[:, 0].min() != 0 or not 0 < reference_mdot <= data[:, 0].max():
            raise ValueError("Reference points must include zero flow and cover reference_mdot")
        maximum = float(data[:, 0].max()) * mdot / reference_mdot
        coefficients = np.polyfit(data[:, 0], data[:, 1], 2)
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (3,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("Pump coefficients must be three finite numbers [a, b, c]")
    a, b, c = map(float, coefficients)
    reference_pressure = (a * reference_mdot + b) * reference_mdot + c
    if a >= 0 or b > 0 or c <= 0 or reference_pressure <= 0:
        raise ValueError("Pump curve must decrease for positive flow, be concave down and positive at the reference point")
    horizontal = reference_mdot / mdot
    vertical = dP / reference_pressure
    a, b, c = a * vertical * horizontal**2, b * vertical * horizontal, c * vertical
    zero_pressure_flow = 2 * c / (sqrt(b*b - 4*a*c) - b)
    curve = PumpCurve(a, b, c, min(maximum, zero_pressure_flow))
    if not all(isfinite(x) for x in (a, b, c, curve.max_mdot)):
        raise ValueError("Scaled pump curve coefficients overflowed")
    curve.validate_flow(mdot)
    return curve
