import numpy as np
import matproplib as mp
from .Section import Section
from ..utils import distribute as dist
from ..utils import aero
from ..utils import geometry as geo
from dataclasses import dataclass

@dataclass
class NoseconeInputs:
    fineness_ratio: float
    profile: float
    reco_mass: float
    material: str

class Nosecone(Section):

    def __init__(self, cfg: dict):

        super().__init__(cfg)
        self.OMLD = cfg["vehicle"]["OMLD"]
        self.fineness_ratio = cfg["nosecone"]["fineness_ratio"]
        self.profile = cfg["nosecone"]["profile"]
        self.L_total = self.OMLD * self.fineness_ratio
        self.length = self.L_total - cfg["avi_bay"]["length"]
        self.n = int(np.ceil(self.length / self.dx))

    def get_mass(self):
        reco_mass = self.cfg["nosecone"]["reco_mass"]
        self.mass = self._get_shell_mass() + dist.uniform(reco_mass, self.n)

    def _get_shell_mass(self) -> np.ndarray:
        x = (np.arange(self.n) + 0.5) * self.dx
        self.radius = self._get_profile(x)
        self.wall_thickness = 0.0032
        self.surf_area = 2 * np.pi * self.radius * self.dx
        V = self.surf_area * self.wall_thickness
        return mp.db.get_material("carbon_fiber_standard").get("density") * V

    def get_EI(self):
        x = (np.arange(self.n) + 0.5) * self.dx
        r_o = self._get_profile(x)
        r_i = np.maximum(r_o - self.wall_thickness, 0.0)
        E = mp.db.get_material("carbon_fiber_standard").get("elastic_modulus_0deg", 300.0)
        self.EI = E * geo.annulus_second_moment(r_o, r_i)

    def get_area(self):
        self.lat_area = 2 * self.radius * self.dx

    def get_MOI(self):
        self.cg = np.sum(self.mass * self.station) / np.sum(self.mass)
        self.Ixx = np.sum(self.mass * self.radius**2)
        self.Iyy = np.sum(self.mass * (self.station - self.cg)**2)

    def get_CNa(self, M: float, alpha: float):
        n_avi = int(np.ceil(self.cfg["avi_bay"]["length"] / self.dx))
        x_avi = self.length + (np.arange(n_avi) + 0.5) * self.dx
        lat_area_avi = np.sum(2 * self._get_profile(x_avi) * self.dx)
        lat_area_total = np.sum(self.lat_area) + lat_area_avi
        self.CNa = aero.nosecone_CNa(M) * self.lat_area / lat_area_total

    def _get_profile(self, x: np.ndarray) -> np.ndarray:
        R = self.OMLD * 0.5
        if self.profile == "von_karman":
            return geo.vk_profile(x, self.L_total, R)
        elif self.profile == "power_series":
            n = self.cfg["nosecone"].get("power_series_n", 0.66)
            return geo.power_series_profile(x, self.L_total, R, n)
        raise ValueError(f"Unknown nosecone profile: {self.profile}")