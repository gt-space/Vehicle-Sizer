from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.interpolate import PchipInterpolator

from AeroTables import DragModel
from simulation_types import AeroOut, AtmosState, KinematicsState


INCH = 0.0254


class Aero:
    """Evaluate one vehicle against an already-loaded RASAero model."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        candidate: Dict[str, float],
        model: DragModel,
    ) -> None:
        schedule = np.asarray(cfg["aoa_schedule"], dtype=float)
        if schedule.ndim != 2 or schedule.shape[1] != 2 or len(schedule) < 2:
            raise ValueError("aoa_schedule requires at least two [time, aoa_deg] rows")
        if not np.all(np.isfinite(schedule)) or np.any(np.diff(schedule[:, 0]) <= 0.0):
            raise ValueError("AoA schedule times must be finite and strictly increasing")

        self.model = model
        self.candidate = {name: float(value) for name, value in candidate.items()}
        model.check(self.candidate)
        self.nose = str(cfg.get("nose", "vonkarman"))
        self.finish = str(cfg.get("finish", "10um"))
        self.fins_on_boattail = bool(cfg.get("fins_on_boattail", True))
        self.schedule_time = schedule[:, 0]
        self.schedule_alpha = np.deg2rad(schedule[:, 1])
        self.reference_area = np.pi * (self.candidate["omld"] * INCH) ** 2 / 4.0

        # Build this candidate's surfaces once. Flight-loop calls only interpolate.
        cd_off = model.cd_table(self.candidate, self.nose, self.finish)
        cd_on = model.cd_table(
            self.candidate,
            self.nose,
            self.finish,
            power_on=True,
        )
        cn = model.cn_table(self.candidate, self.nose)
        cp = model.cp_table(
            self.candidate,
            self.nose,
            fins_on_boattail=self.fins_on_boattail,
        )
        self.mach = np.asarray(model.mach, dtype=float)
        self.alpha_deg = np.asarray(model.alpha, dtype=float)
        self._cd = {
            False: PchipInterpolator(self.alpha_deg, cd_off, axis=1),
            True: PchipInterpolator(self.alpha_deg, cd_on, axis=1),
        }
        self._cn = PchipInterpolator(self.alpha_deg, cn, axis=1)
        self._cp = PchipInterpolator(self.alpha_deg, cp, axis=1)

    def aoa(self, time: float) -> float:
        """Return scheduled angle of attack in radians."""

        if time < self.schedule_time[0] or time > self.schedule_time[-1]:
            raise ValueError("Flight time is outside the AoA schedule")
        return float(np.interp(time, self.schedule_time, self.schedule_alpha))

    def _coordinates(self, mach: float, alpha: float) -> tuple[float, float, float]:
        if not np.isfinite(mach) or not np.isfinite(alpha):
            raise ValueError("Mach and angle of attack must be finite")
        mach = abs(float(mach))
        if mach > self.mach[-1]:
            raise ValueError(f"Mach must not exceed {self.mach[-1]}")
        # Hold the first coefficient below Mach 0.1 while q tends to zero.
        mach = max(mach, float(self.mach[0]))
        alpha_sign = float(np.sign(alpha))
        alpha_deg = abs(float(np.degrees(alpha)))
        if alpha_deg > self.alpha_deg[-1]:
            raise ValueError(f"Angle of attack must not exceed {self.alpha_deg[-1]} deg")
        return mach, alpha_deg, alpha_sign

    def _at(self, interpolator: PchipInterpolator, mach: float, alpha_deg: float) -> float:
        return float(np.interp(mach, self.mach, interpolator(alpha_deg)))

    def coefficients(
        self,
        mach: float,
        alpha: float,
        engine_on: bool,
    ) -> tuple[float, float, float, float]:
        """Return wind-axis CD, body-axis CA, signed CN, and CP in metres."""

        mach, alpha_deg, alpha_sign = self._coordinates(mach, alpha)
        cd = self._at(self._cd[engine_on], mach, alpha_deg)
        cn_magnitude = self._at(self._cn, mach, alpha_deg)
        cn = alpha_sign * cn_magnitude
        alpha_rad = np.deg2rad(alpha_deg)
        ca = (cd - cn_magnitude * np.sin(alpha_rad)) / np.cos(alpha_rad)
        cp = self._at(self._cp, mach, alpha_deg) * INCH
        return cd, ca, cn, cp

    def normal_distribution(self, mach: float, alpha: float) -> Dict[str, Any]:
        """Return signed dCN/dx on SI stations for structural loads."""

        mach, alpha_deg, alpha_sign = self._coordinates(mach, alpha)
        output = self.model.cn_distribution(
            self.candidate,
            mach,
            alpha_deg,
            nose=self.nose,
            finish=self.finish,
            fins_on_boattail=self.fins_on_boattail,
        )
        return {
            **output,
            "x": np.asarray(output["x"], dtype=float) * INCH,
            "dcn_dx": alpha_sign
            * np.asarray(output["dcn_dx"], dtype=float)
            / INCH,
            "cn": alpha_sign * float(output["cn"]),
            "cp": float(output["cp"]) * INCH,
        }

    def axial_distribution(self, mach: float, alpha: float, engine_on: bool) -> Dict[str, Any]:
        """Return dCA/dx in SI units, including the reader's base spike.

        point_loads describes contributions already present in dca_dx/parts;
        consumers must remove their spikes before adding explicit point loads.
        CA is even in alpha; unlike CN it does not acquire alpha's sign.
        """
        mach, alpha_deg, _ = self._coordinates(mach, alpha)
        output = self.model.ca_distribution(
            self.candidate, mach, alpha_deg, nose=self.nose, finish=self.finish,
            fins_on_boattail=self.fins_on_boattail, power_on=engine_on,
        )
        return {
            **output,
            "x": np.asarray(output["x"], dtype=float) * INCH,
            "diameter": np.asarray(output["diameter"], dtype=float) * INCH,
            "dca_dx": np.asarray(output["dca_dx"], dtype=float) / INCH,
            "parts": {key: np.asarray(value, dtype=float) / INCH
                      for key, value in output["parts"].items()},
            "point_loads": {key: (float(station) * INCH, float(coefficient))
                            for key, (station, coefficient) in output["point_loads"].items()},
        }

    def evaluate(
        self,
        kin: KinematicsState,
        atm: AtmosState,
        engine_on: bool,
    ) -> AeroOut:
        """Return aerodynamic coefficients and forces at the current state."""

        cd, ca, cn, cp = self.coefficients(atm.Ma, kin.alpha, engine_on)
        scale = atm.q * self.reference_area
        return AeroOut(
            Cd=cd,
            D=scale * cd,
            Ca=ca,
            A=scale * ca,
            Cn=cn,
            N=scale * cn,
            cp=cp,
        )


def gravity(m: float, h: float) -> float:
    g0 = 9.80665
    Re = 6378137
    return m * g0 * (Re / (Re + h))**2
