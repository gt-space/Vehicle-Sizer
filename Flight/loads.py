from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

class Loads:

    def __init__(self, vehicle, aero):
        self.vehicle = vehicle
        self.aero = aero
        self.ref_area = aero.reference_area

    def _cell_coefficients(self, x, density):
        """Integrate piecewise-linear coefficient density over structural cells."""
        x, density = np.asarray(x, dtype=float), np.asarray(density, dtype=float)
        edges = self.vehicle.cell_edges
        if (x.ndim != 1 or len(x) < 2 or density.shape != x.shape
                or not np.all(np.isfinite(x)) or not np.all(np.isfinite(density))
                or np.any(np.diff(x) <= 0)):
            raise ValueError("Aero distribution requires finite densities and increasing stations")
        if not np.allclose([x[0], x[-1]], [edges[0], edges[-1]], rtol=0, atol=1e-8):
            raise ValueError("Aero distribution must cover the actual vehicle extent")
        running = np.r_[0.0, np.cumsum(.5 * (density[1:] + density[:-1]) * np.diff(x))]
        index = np.clip(np.searchsorted(x, edges, side="right") - 1, 0, len(x)-2)
        offset = np.clip(edges, x[0], x[-1]) - x[index]
        slope = np.diff(density) / np.diff(x)
        cumulative = running[index] + density[index]*offset + .5*slope[index]*offset**2
        return np.diff(cumulative)

    def get_axial_forces(self, q: float, mach: float, alpha: float, engine_on: bool) -> np.ndarray:
        """Aerodynamic force per cell, with base drag applied exactly once aft."""
        distribution = self.aero.axial_distribution(mach, alpha, engine_on)
        density = np.asarray(distribution["dca_dx"], dtype=float).copy()
        points = distribution["point_loads"]
        for key in points:
            density -= distribution["parts"][key]
        coefficients = self._cell_coefficients(distribution["x"], density)
        for key, (station, coefficient) in points.items():
            if key != "base" or not np.isclose(station, self.vehicle.cell_edges[-1], rtol=0, atol=1e-8):
                raise ValueError("Axial aero point load must be the base load at the aft boundary")
            coefficients[-1] += coefficient
        if not np.isclose(coefficients.sum(), distribution["ca"], rtol=1e-8, atol=1e-10):
            raise ValueError("Mapped axial distribution does not conserve total CA")
        return q * self.ref_area * coefficients

    def get_axial_load(self, axial_forces: np.ndarray, thrust: float) -> np.ndarray:
        """Return body-axis internal load from table CA and engine thrust.

        Positive station points nose-to-aft. Aerodynamic axial force acts aft
        and thrust acts forward; the returned sign is positive in compression.
        The engine's forward end is the assumed thrust/airframe interface.
        Its point load is assigned to the cell immediately aft of that boundary.
        """

        v = self.vehicle
        force = np.asarray(axial_forces, dtype=float).copy()
        if force.shape != v.station.shape or not np.all(np.isfinite(force)):
            raise ValueError("Axial aerodynamic forces must have one finite value per vehicle cell")
        interface = float(v.engine_start_station)
        if not np.isfinite(interface) or not v.cell_edges[0] <= interface < v.cell_edges[-1]:
            raise ValueError("Engine thrust interface must lie within the vehicle grid")
        thrust_cell = np.searchsorted(v.cell_edges, interface, side="right") - 1
        force[thrust_cell] -= float(thrust)
        acceleration = np.sum(force) / v.total_mass
        effective_force = force - v.mass * acceleration
        return np.cumsum(effective_force)

    def get_normal_load(self, q: float, M: float, alpha: float):

        v = self.vehicle
        x = v.station

        distribution = self.aero.normal_distribution(M, alpha)
        cell_cn = self._cell_coefficients(distribution["x"], distribution["dcn_dx"])
        N = q * self.ref_area * cell_cn

        a_trans = np.sum(N) / v.total_mass
        r = x - v.cg
        a_ang = np.sum(N * r) / v.Iyy

        L1 = v.mass * a_trans
        L2 = a_ang * r * v.mass

        N = N - L1 - L2
        return N

    def evaluate(
        self,
        q: float,
        mach: float,
        alpha: float,
        axial_force: float,
        thrust: float,
        engine_on: bool,
    ) -> dict[str, np.ndarray]:
        """Evaluate synchronized aerodynamic, inertial, and internal loads."""

        axial_aero = self.get_axial_forces(q, mach, alpha, engine_on)
        if not np.isclose(axial_aero.sum(), axial_force, rtol=1e-7, atol=1e-7):
            raise ValueError("Distributed axial force disagrees with flight CA force")
        axial = self.get_axial_load(axial_aero, thrust)
        normal = self.get_normal_load(q, mach, alpha)
        shear, bending, _, _ = self.beam_deflection(normal)
        return {
            "station": self.vehicle.station.copy(),
            "axial": axial,
            "axial_aero": axial_aero,
            "normal": normal,
            "shear": shear,
            "bending": bending,
        }

    def beam_deflection(self, N):

        v = self.vehicle
        EI = v.EI
        x = v.station
        widths = np.diff(getattr(v, "cell_edges", np.concatenate((x, [v.length]))))
        V = np.cumsum(N)
        M = np.cumsum(V * widths)
        kappa = -M / EI
        theta = cumulative_trapezoid(kappa, x=x, initial=0)

        idx_cg = np.argmin(np.abs(x - v.cg))
        theta = theta - theta[idx_cg]

        nu = cumulative_trapezoid(theta, x=x, initial=0)
        nu = nu - nu[idx_cg]

        return V, M, theta, nu
