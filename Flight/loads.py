from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from Vehicle.sections.Nosecone import Nosecone
from Vehicle.sections.FinCan import FinCan


class Loads:

    def __init__(self, vehicle, ref_area: float):
        self.vehicle = vehicle
        self.ref_area = ref_area

    def get_axial_load(self, D: float, T: float) -> np.ndarray:

        v = self.vehicle
        x = v.station

        fN, fB, fF = 0.2, 0.5, 0.3

        F = np.zeros_like(x)
        idx_nose = np.zeros(x.shape, dtype=bool)
        idx_body = np.zeros(x.shape, dtype=bool)
        idx_fins = np.zeros(x.shape, dtype=bool)

        for s in v.sections:
            idx = (x >= s.start_station) & (x < s.end_station)
            if isinstance(s, Nosecone):
                idx_nose |= idx
            elif isinstance(s, FinCan):
                idx_fins |= idx
            else:
                idx_body |= idx

        a_surf = v.surf_area
        F[idx_nose] = D * fN * a_surf[idx_nose] / np.sum(a_surf[idx_nose])
        F[idx_body] = D * fB * a_surf[idx_body] / np.sum(a_surf[idx_body])
        F[idx_fins] = D * fF * a_surf[idx_fins] / np.sum(a_surf[idx_fins])

        iE = x.size - 1
        F[iE] += T

        a_ax = np.sum(F) / v.total_mass
        F = F - v.mass * a_ax

        P = cumulative_trapezoid(-F, initial=0)
        return P

    def get_normal_load(self, q: float, M: float, alpha: float):

        v = self.vehicle
        x = v.station

        v.get_CNa(M, alpha)
        CNa = v.CNa

        N = q * CNa * self.ref_area

        a_trans = np.sum(N) / v.total_mass
        r = x - v.cg
        a_ang = np.sum(N * r) / v.Iyy

        L1 = v.mass * a_trans
        L2 = a_ang * r * v.mass

        N = N - L1 - L2
        return N

    def beam_deflection(self, N):

        v = self.vehicle
        EI = v.EI
        x = v.station
        dx = x[1] - x[0]

        V = cumulative_trapezoid(N, dx=dx, initial=0)
        M = cumulative_trapezoid(V, dx=dx, initial=0)
        kappa = -M / EI
        theta = cumulative_trapezoid(kappa, dx=dx, initial=0)

        idx_cg = np.argmin(np.abs(x - v.cg))
        theta = theta - theta[idx_cg]

        nu = cumulative_trapezoid(theta, dx=dx, initial=0)
        nu = nu - nu[idx_cg]

        return V, M, theta, nu