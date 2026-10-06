import numpy as np
from ..Material import mp
from .Section import Section
from ..utils import distribute as dist
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
        self.surf_area = 2 * np.pi * self.radius * self.dx
        self.shell_mass = np.zeros(self.n)
        self.mass = np.zeros(self.n)

    def get_EI(self):
        self.EI = np.zeros(self.n)

    def get_area(self):
        self.lat_area = 2 * self.radius * self.dx

    def get_MOI(self):
        self.cg = (np.sum(self.mass * self.station) / np.sum(self.mass)
                   if np.sum(self.mass) > 0 else np.mean(self.station))
        self.Ixx = np.sum(self.mass * self.radius**2)
        self.Iyy = np.sum(self.mass * (self.station - self.cg)**2)


    def get_thermal_oml_area(self) -> np.ndarray:
        return self.surf_area.copy()

    def get_thermal_shell_mass(self) -> np.ndarray:
        return self.shell_mass.copy()
