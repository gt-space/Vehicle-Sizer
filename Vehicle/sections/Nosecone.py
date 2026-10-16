import numpy as np
from ..Material import mp
from .Section import Section
from ..utils import geometry as geo


class Nosecone(Section):

    def __init__(self, cfg: dict):

        super().__init__(cfg)
        self.OMLD = cfg["vehicle"]["OMLD"]
        self.fineness_ratio = cfg["nosecone"]["fineness_ratio"]
        self.L_total = self.OMLD * self.fineness_ratio
        self.length = self.L_total - cfg["avi_bay"]["length"]
        self.set_grid()
        self.wall_material = cfg["nosecone"]["material"]
        self.wall_thickness = float(cfg["nosecone"]["wall_thickness"])
        if not np.isfinite(self.wall_thickness) or self.wall_thickness <= 0:
            raise ValueError("Nosecone wall_thickness must be finite and positive")
        self.emissivity = 0.85
        self.radius = geo.nose_profile(self.local_centers, self.L_total, self.OMLD * .5, cfg["nosecone"])

    def get_mass(self):
        self.shell_mass = self._get_shell_mass()
        self.mass = self.shell_mass.copy()

    def _get_shell_mass(self) -> np.ndarray:
        self.surf_area = 2 * np.pi * self.radius * self.dx
        V = self.surf_area * self.wall_thickness
        return mp.get_material(self.wall_material).get("density") * V

    def get_EI(self):
        r_o = self.radius
        r_i = np.maximum(r_o - self.wall_thickness, 0.0)
        E = mp.get_material(self.wall_material).get("elastic_modulus", T=300.0)
        self.EI = E * geo.annulus_second_moment(r_o, r_i)

    def get_area(self):
        self.lat_area = 2 * self.radius * self.dx
