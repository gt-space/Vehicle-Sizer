"""Atmosphere and wind models for the planar flight simulation.

The US Standard Atmosphere 1976 is tabulated once and interpolated for fast
per-timestep lookups. Wind profiles are loaded once from CSV and linearly
interpolated with altitude.
"""
from pathlib import Path
import csv

import numpy as np
import ussa1976

from simulation_types import AtmosState


R_AIR = 287.05287    # J/(kg K)
GAMMA = 1.4

WIND_COLUMNS = ("altitude_m", "wind_x_m_s", "wind_z_m_s")


def sutherland(T):
    return 1.458e-6 * T**1.5 / (T + 110.4)


class WindProfile:
    """
    This wrapper reads the wind profile csv.
    Altitude-dependent horizontal and vertical wind components.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

        if not self.path.is_file():
            raise ValueError(f"Wind profile file does not exist: {self.path}")

        try:
            with self.path.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)

                if reader.fieldnames != list(WIND_COLUMNS):
                    raise ValueError(
                        "Wind profile must have exactly these columns: "
                        + ",".join(WIND_COLUMNS)
                    )

                rows = list(reader)
        except OSError as exc:
            raise ValueError(f"Could not read wind profile: {self.path}") from exc

        if len(rows) < 2:
            raise ValueError("Wind profile requires at least two data rows")

        try:
            altitude = np.asarray(
                [float(row["altitude_m"]) for row in rows],
                dtype=float,
            )
            wind_x = np.asarray(
                [float(row["wind_x_m_s"]) for row in rows],
                dtype=float,
            )
            wind_z = np.asarray(
                [float(row["wind_z_m_s"]) for row in rows],
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Wind profile values must be numeric") from exc

        if not (
            np.all(np.isfinite(altitude))
            and np.all(np.isfinite(wind_x))
            and np.all(np.isfinite(wind_z))
        ):
            raise ValueError("Wind profile values must be finite")

        if np.any(np.diff(altitude) <= 0.0):
            raise ValueError(
                "Wind profile altitude_m values must be strictly increasing "
                "with no duplicates"
            )

        self.altitude = altitude
        self.wind_x = wind_x
        self.wind_z = wind_z

    def wind(self, altitude: float) -> tuple[float, float]:
        """Return wind components at altitude, or zero outside the profile."""

        if not np.isfinite(altitude):
            raise ValueError("Wind lookup altitude must be finite")

        wind_x = np.interp(
            altitude,
            self.altitude,
            self.wind_x,
            left=0.0,
            right=0.0,
        )
        wind_z = np.interp(
            altitude,
            self.altitude,
            self.wind_z,
            left=0.0,
            right=0.0,
        )
        return float(wind_x), float(wind_z)


class Environment:

    def __init__(
        self,
        h_max: float = 150e3,
        dh: float = 100.0,
        wind_profile: str | Path | None = None,
    ):
        self._z = np.arange(0.0, h_max + dh, dh)
        ds = ussa1976.compute(z=self._z, variables=["t", "p", "rho"])
        self._T = ds["t"].values
        # p and rho span ~10 decades; interpolate in log space
        self._log_p = np.log(ds["p"].values)
        self._log_rho = np.log(ds["rho"].values)

        self._wind = WindProfile(wind_profile) if wind_profile is not None else None

    def wind(self, altitude: float) -> tuple[float, float]:
        """Return wind in the simulation +x and +z directions."""

        if self._wind is None:
            return 0.0, 0.0
        return self._wind.wind(altitude)

    def atmosphere(self, h: float, airspeed: float) -> AtmosState:
        T = np.interp(h, self._z, self._T)
        p = np.exp(np.interp(h, self._z, self._log_p))
        rho = np.exp(np.interp(h, self._z, self._log_rho))
        a = np.sqrt(GAMMA * R_AIR * T)
        mu = sutherland(T)
        Ma = abs(airspeed) / a
        q = 0.5 * rho * airspeed**2
        return AtmosState(T=float(T), p=float(p), rho=float(rho), mu=float(mu),
                          a=float(a), q=float(q), Ma=float(Ma))
