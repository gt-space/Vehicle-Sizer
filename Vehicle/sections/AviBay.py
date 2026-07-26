import numpy as np
from scipy.optimize import root_scalar
import matproplib as mp
from .Section import Section
from ..utils import distribute as dist
from ..utils import aero
from ..utils import geometry as geo

class AviBay(Section):

    def __init__(self, cfg: dict):

        super().__init__(cfg)
        self.OMLD = cfg["vehicle"]["OMLD"]
        self.length = cfg["avi_bay"]["length"]
        self.n = int(np.ceil(self.length / self.dx))

        L_total = self.OMLD * cfg["nosecone"]["fineness_ratio"]
        L_nosecone = L_total - self.length
        R = self.OMLD * 0.5
        x = L_nosecone + (np.arange(self.n) + 0.5) * self.dx
        profile = cfg["nosecone"]["profile"]
        if profile == "von_karman":
            self.r_entry = float(geo.vk_profile(np.array([L_nosecone]), L_total, R)[0])
            self.radius = geo.vk_profile(x, L_total, R)
        elif profile == "power_series":
            n = cfg["nosecone"].get("power_series_n", 0.66)
            self.r_entry = float(geo.power_series_profile(np.array([L_nosecone]), L_total, R, n)[0])
            self.radius = geo.power_series_profile(x, L_total, R, n)
        else:
            raise ValueError(f"Unknown nosecone profile: {profile}")

    def get_mass(self):
        lump_masses = self._get_bulkhead_mass() + self.cfg["avi_bay"]["avi_mass"]
        self.mass = self._get_shell_mass() + dist.uniform(lump_masses, self.n)

    def _get_shell_mass(self) -> np.ndarray:
        mat = mp.db.get_material(self.cfg["avi_bay"]["clamshell_material"])
        t = self.cfg["avi_bay"]["clamshell_thickness"]
        rho = mat.get("density")
        self.surf_area = 2 * np.pi * self.radius * self.dx
        return rho * self.surf_area * t

    def _get_bulkhead_mass(self) -> float:
        a = 9.81 * 10.0
        P = self.cfg["avi_bay"]["avi_mass"] * a

        mat = mp.db.get_material(self.cfg["avi_bay"]["bulkhead_material"])
        nu = mat.get("poisson_ratio_major")
        T = 350.0
        sigma = mat.get("yield_strength_90deg", T)

        r = (self.OMLD * 0.5) - self.cfg["avi_bay"]["clamshell_thickness"]
        t = self._get_bulkhead_thickness(P, r, nu, sigma)
        rho = mat.get("density")
        return np.pi * r**2 * t * rho

    def _get_bulkhead_thickness(self, P, r, nu, sigma) -> float:
        FOS = 1.5

        def f(t):
            return (P / t**2) * (1 + nu) * (0.485 * np.log(r / t) + 0.52) - sigma

        t_bounds = (1e-3, 0.1)
        sol = root_scalar(f, bracket=t_bounds, method='brentq')
        return sol.root * FOS

    def get_EI(self):
        t = self.cfg["avi_bay"]["clamshell_thickness"]
        r_i = np.maximum(self.radius - t, 0.0)
        mat = mp.db.get_material(self.cfg["avi_bay"]["clamshell_material"])
        E = mat.get("elastic_modulus_0deg") if "elastic_modulus_0deg" in mat.properties else mat.get("elastic_modulus")
        self.EI = E * geo.annulus_second_moment(self.radius, r_i)

    def get_area(self):
        self.lat_area = 2 * self.radius * self.dx

    def get_MOI(self):
        self.cg = np.sum(self.mass * self.station) / np.sum(self.mass)
        self.Ixx = np.sum(self.mass * self.radius**2)
        self.Iyy = np.sum(self.mass * (self.station - self.cg)**2)

    def get_CNa(self, M: float, alpha: float):
        R = self.OMLD * 0.5
        L_total = self.OMLD * self.cfg["nosecone"]["fineness_ratio"]
        L_nosecone = L_total - self.length
        n_nose = int(np.ceil(L_nosecone / self.dx))
        x_nose = (np.arange(n_nose) + 0.5) * self.dx
        profile = self.cfg["nosecone"]["profile"]
        if profile == "von_karman":
            r_nose = geo.vk_profile(x_nose, L_total, R)
        else:
            n_ps = self.cfg["nosecone"].get("power_series_n", 0.66)
            r_nose = geo.power_series_profile(x_nose, L_total, R, n_ps)
        lat_area_total = np.sum(2 * r_nose * self.dx) + np.sum(self.lat_area)
        self.CNa = aero.nosecone_CNa(M) * self.lat_area / lat_area_total