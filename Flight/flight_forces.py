from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .types import AeroOut, AtmosState, KinematicsState


class Aero:
    """Tabular drag model interpolated in Mach and angle of attack."""

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

        mach, alpha, cd_on, cd_off = self._load_deck(
            Path(cfg["cd_table"]), str(cfg["stratum"])
        )
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

    @staticmethod
    def _load_deck(
        path: Path, stratum: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load one vehicle design from an aerodynamic HDF5 stratum."""

        with h5py.File(path, "r") as deck:
            required = ("mach", "alpha", "strata")
            if any(name not in deck for name in required):
                raise ValueError(f"Aerodynamic deck requires datasets {required}")
            if stratum not in deck["strata"]:
                raise ValueError(f"Aerodynamic stratum {stratum!r} does not exist")

            group = deck["strata"][stratum]
            required_coefficients = ("cd_on", "cd_wind")
            if any(name not in group for name in required_coefficients):
                raise ValueError(
                    f"Aerodynamic stratum requires datasets {required_coefficients}"
                )

            mach = np.asarray(deck["mach"], dtype=float)
            alpha_deg = np.asarray(deck["alpha"], dtype=float)
            cd_on = np.asarray(group["cd_on"], dtype=float)
            cd_off = np.asarray(group["cd_wind"], dtype=float)

        if mach.ndim != 1 or alpha_deg.ndim != 1:
            raise ValueError("Aerodynamic Mach and alpha axes must be one-dimensional")
        if len(mach) < 2 or len(alpha_deg) < 2:
            raise ValueError("Aerodynamic deck requires at least two Mach and alpha values")
        if not np.all(np.isfinite(mach)) or not np.all(np.isfinite(alpha_deg)):
            raise ValueError("Aerodynamic axes must be finite")
        if np.any(mach < 0.0) or np.any(np.diff(mach) <= 0.0):
            raise ValueError("Aerodynamic Mach values must be nonnegative and increasing")
        if np.any(np.diff(alpha_deg) <= 0.0):
            raise ValueError("Aerodynamic alpha values must be strictly increasing")

        expected = (1, len(mach), len(alpha_deg))
        if cd_on.shape != expected or cd_off.shape != expected:
            raise ValueError(
                f"Aerodynamic coefficient tables must have shape {expected}"
            )
        cd_on = cd_on[0]
        cd_off = cd_off[0]
        if not np.all(np.isfinite(cd_on)) or not np.all(np.isfinite(cd_off)):
            raise ValueError("Aerodynamic coefficients must be finite")
        if np.any(cd_on < 0.0) or np.any(cd_off < 0.0):
            raise ValueError("Aerodynamic coefficients cannot be negative")

        return mach, np.deg2rad(alpha_deg), cd_on, cd_off

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


#move these elsewhere? 
def drag(Cd:float, q: float, A_ref: float) -> float:
    return Cd * q * A_ref

def gravity(m: float, h: float) -> float:
    g0 = 9.80665
    Re = 6378137
    return m * g0 * (Re / (Re + h))**2
