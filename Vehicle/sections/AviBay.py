import numpy as np
from .Section import Section
from ..utils import geometry as geo

class AviBay(Section):

    def __init__(self, cfg: dict):

        super().__init__(cfg)
        self.OMLD = cfg["vehicle"]["OMLD"]
        self.length = cfg["avi_bay"]["length"]
        self.set_grid()
        self.wall_thickness = 0.0
        self.wall_material = cfg["nosecone"]["material"]
        self.emissivity = 0.85

        L_total = self.OMLD * cfg["nosecone"]["fineness_ratio"]
        L_nosecone = L_total - self.length
        self.radius = geo.nose_profile(L_nosecone + self.local_centers, L_total,
                                       self.OMLD * .5, cfg["nosecone"])

    def get_mass(self):
        self.surf_area = 2 * np.pi * self.radius * self.dx
        self.shell_mass = np.zeros(self.n)
        self.mass = np.zeros(self.n)

    def get_EI(self):
        self.EI = np.zeros(self.n)

    def get_area(self):
        self.lat_area = 2 * self.radius * self.dx
