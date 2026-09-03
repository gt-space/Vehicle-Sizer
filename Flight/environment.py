"""Atmosphere model: US Standard Atmosphere 1976 (full 0-1000 km standard via
the `ussa1976` package), tabulated once and interpolated for fast per-timestep
lookups. `Environment.atmosphere(h, v)` returns the freestream AtmosState."""
import numpy as np
import ussa1976
from .types import AtmosState

R_AIR = 287.05287    # J/(kg K)
GAMMA = 1.4


def sutherland(T):
    return 1.458e-6 * T**1.5 / (T + 110.4)

class Environment:

    def __init__(self, h_max: float = 150e3, dh: float = 100.0):
        self._z = np.arange(0.0, h_max + dh, dh)
        ds = ussa1976.compute(z=self._z, variables=["t", "p", "rho"])
        self._T = ds["t"].values
        # p and rho span ~10 decades; interpolate in log space
        self._log_p = np.log(ds["p"].values)
        self._log_rho = np.log(ds["rho"].values)

    def atmosphere(self, h: float, v: float) -> AtmosState:
        z = np.clip(h, self._z[0], self._z[-1])
        T = np.interp(z, self._z, self._T)
        p = np.exp(np.interp(z, self._z, self._log_p))
        rho = np.exp(np.interp(z, self._z, self._log_rho))
        a = np.sqrt(GAMMA * R_AIR * T)
        mu = sutherland(T)
        Ma = abs(v) / a
        q = 0.5 * rho * v**2
        return AtmosState(T=float(T), p=float(p), rho=float(rho), mu=float(mu),
                          a=float(a), q=float(q), Ma=float(Ma))
