import numpy as np
from ..Material import mp
from .Section import Section
from ..utils import distribute as dist
from ..utils import geometry as geo


class Nosecone(Section):

    def __init__(self, cfg: dict):

        super().__init__(cfg)
        self.OMLD = cfg["vehicle"]["OMLD"]
        self.fineness_ratio = cfg["nosecone"]["fineness_ratio"]
        self.profile = cfg["nosecone"]["profile"]
        self.L_total = self.OMLD * self.fineness_ratio
        self.length = self.L_total - cfg["avi_bay"]["length"]
        self.set_grid()
        self.wall_material = cfg["nosecone"]["material"]
        self.wall_thickness = float(cfg["nosecone"]["wall_thickness"])
        if not np.isfinite(self.wall_thickness) or self.wall_thickness <= 0:
            raise ValueError("Nosecone wall_thickness must be finite and positive")
        self.emissivity = 0.85

    def get_mass(self):
        self.shell_mass = self._get_shell_mass()
        self.mass = self.shell_mass.copy()

    def _get_shell_mass(self) -> np.ndarray:
        x = (np.arange(self.n) + 0.5) * self.dx
        self.radius = self._get_profile(x)
        self.surf_area = 2 * np.pi * self.radius * self.dx
        V = self.surf_area * self.wall_thickness
        return mp.get_material(self.wall_material).get("density") * V

    def get_EI(self):
        x = (np.arange(self.n) + 0.5) * self.dx
        r_o = self._get_profile(x)
        r_i = np.maximum(r_o - self.wall_thickness, 0.0)
        E = mp.get_material(self.wall_material).get("elastic_modulus", T=300.0)
        self.EI = E * geo.annulus_second_moment(r_o, r_i)

    def get_area(self):
        self.lat_area = 2 * self.radius * self.dx

    def get_MOI(self):
        self.cg = np.sum(self.mass * self.station) / np.sum(self.mass)
        self.Ixx = np.sum(self.mass * self.radius**2)
        self.Iyy = np.sum(self.mass * (self.station - self.cg)**2)


    def get_thermal_oml_area(self) -> np.ndarray:
        return self.surf_area.copy()

    def get_thermal_shell_mass(self) -> np.ndarray:
        return self.shell_mass.copy()

    def _get_profile(self, x: np.ndarray) -> np.ndarray:
        R = self.OMLD * 0.5
        if self.profile == "von_karman":
            return geo.vk_profile(x, self.L_total, R)
        elif self.profile == "power_series":
            n = self.cfg["nosecone"].get("power_series_n", 0.66)
            return geo.power_series_profile(x, self.L_total, R, n)
        raise ValueError(f"Unknown nosecone profile: {self.profile}")
