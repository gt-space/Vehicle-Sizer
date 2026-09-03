from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .types import AeroOut, AtmosState, KinematicsState


class Aero:
    """Tabular drag model interpolated in Mach and angle of attack."""

    columns = {"mach", "aoa_deg", "cd_engine_on", "cd_engine_off"}

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.reference_area = float(cfg["reference_area"])
        if self.reference_area <= 0.0:
            raise ValueError("Aerodynamic reference area must be positive")

        schedule = np.asarray(cfg["aoa_schedule"], dtype=float)
        if schedule.ndim != 2 or schedule.shape[1] != 2 or len(schedule) < 2:
            raise ValueError("aoa_schedule requires at least two [time, aoa_deg] rows")
        if not np.all(np.isfinite(schedule)) or np.any(np.diff(schedule[:, 0]) <= 0.0):
            raise ValueError("AoA schedule times must be finite and strictly increasing")
        self.schedule_time = schedule[:, 0]
        self.schedule_alpha = np.deg2rad(schedule[:, 1])

        mach, alpha, cd_on, cd_off = self._load_deck(Path(cfg["cd_table"]))
        self.mach = mach
        self.alpha = alpha
        self._cd = {
            True: RegularGridInterpolator(
                (mach, alpha), cd_on, method="linear", bounds_error=True
            ),
            False: RegularGridInterpolator(
                (mach, alpha), cd_off, method="linear", bounds_error=True
            ),
        }

    @classmethod
    def _load_deck(cls, path: Path):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not cls.columns.issubset(reader.fieldnames):
                raise ValueError(
                    f"Aerodynamic deck requires columns {sorted(cls.columns)}"
                )
            rows = list(reader)
        if not rows:
            raise ValueError("Aerodynamic deck cannot be empty")

        points = {}
        for row in rows:
            try:
                values = tuple(float(row[name]) for name in cls.columns)
            except (TypeError, ValueError) as error:
                raise ValueError("Aerodynamic deck values must be numeric") from error
            if not np.all(np.isfinite(values)):
                raise ValueError("Aerodynamic deck values must be finite")

            mach = float(row["mach"])
            alpha = np.deg2rad(float(row["aoa_deg"]))
            cd_on = float(row["cd_engine_on"])
            cd_off = float(row["cd_engine_off"])
            if mach < 0.0 or cd_on < 0.0 or cd_off < 0.0:
                raise ValueError("Mach and drag coefficients cannot be negative")
            if (mach, alpha) in points:
                raise ValueError("Aerodynamic deck contains duplicate Mach/AoA rows")
            points[mach, alpha] = (cd_on, cd_off)

        mach_axis = np.array(sorted({point[0] for point in points}))
        alpha_axis = np.array(sorted({point[1] for point in points}))
        if len(mach_axis) < 2 or len(alpha_axis) < 2:
            raise ValueError("Aerodynamic deck requires at least two Mach and AoA values")
        if len(points) != len(mach_axis) * len(alpha_axis):
            raise ValueError("Aerodynamic deck must contain a complete Mach/AoA grid")

        cd_on = np.empty((len(mach_axis), len(alpha_axis)))
        cd_off = np.empty_like(cd_on)
        for i, mach in enumerate(mach_axis):
            for j, alpha in enumerate(alpha_axis):
                try:
                    cd_on[i, j], cd_off[i, j] = points[mach, alpha]
                except KeyError as error:
                    raise ValueError(
                        "Aerodynamic deck must contain a complete Mach/AoA grid"
                    ) from error
        return mach_axis, alpha_axis, cd_on, cd_off

    def aoa(self, time: float) -> float:
        """Return scheduled angle of attack in radians."""

        if time < self.schedule_time[0] or time > self.schedule_time[-1]:
            raise ValueError("Flight time is outside the AoA schedule")
        return float(np.interp(time, self.schedule_time, self.schedule_alpha))

    def cd(self, mach: float, alpha: float, engine_on: bool) -> float:
        """Return the interpolated drag coefficient."""

        return float(self._cd[engine_on]((mach, alpha)))

    def evaluate(
        self,
        kin: KinematicsState,
        atm: AtmosState,
        engine_on: bool,
    ) -> AeroOut:
        """Return drag magnitude for the current flight condition."""

        coefficient = self.cd(abs(atm.Ma), kin.alpha, engine_on)
        return AeroOut(
            Cd=coefficient,
            D=coefficient * atm.q * self.reference_area,
            heat_bc={},
        )

def drag(Cd:float, q: float, A_ref: float) -> float:
    return Cd * q * A_ref

def gravity(m: float, h: float) -> float:
    g0 = 9.80665
    Re = 6378137
    return m * g0 * (Re / (Re + h))**2
