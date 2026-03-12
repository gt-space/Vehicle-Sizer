import numpy as np
import matproplib as mp
from Vehicle.Section import Section

class Nosecone(Section):

    def __init__(self, cfg: dict):
        
        super().__init__(cfg)
        self.OMLD = self.cfg["vehicle"]["OMLD"]
        self.fineness_ratio = 5
        self.length = self.OMLD * self.fineness_ratio
        self.n = int(np.ceil(self.length / self.dx))

    def get_mass(self):
        
        reco_mass = self.cfg["nosecone"]["reco_mass"]
        self.reco_mass = np.full(self.n, reco_mass / self.n)

        shell_mass = self._get_shell_mass()

        self.mass = shell_mass + self.reco_mass

    def _get_shell_mass(self) -> float:

        R = self.OMLD * 0.5
        x = (np.arange(self.n) + 0.5) * self.dx
        self.radius = self._get_vk_profile(x, self.length, R)
        self.surf_area = 2 * np.pi * self.radius * self.dx

        self.wall_thickness = 0.0032
        V = self.surf_area * self.wall_thickness

        mat = mp.db.get_material("carbon_fiber_standard")
        rho = mat.get("density")
        m = rho * V

        return m

    def get_EI(self):

        R = self.OMLD * 0.5
        x = (np.arange(self.n) + 0.5) * self.dx

        r_o = self._get_vk_profile(x, self.length, R)
        r_i = np.maximum(r_o - self.wall_thickness, 0.0)

        I = (np.pi * 0.25) * (r_o**4 - r_i**4)

        T = 300.0
        mat = mp.db.get_material("carbon_fiber_standard")
        E = mat.get("elastic_modulus_0deg", T)

        self.EI = E * I

    def get_area(self):

        R = self.OMLD * 0.5
        x = (np.arange(self.n) + 0.5) * self.dx

        r = self._get_vk_profile(x, self.length, R)

        self.lat_area = 2 * r * self.dx

    def get_CNa(self, M, alpha):
        CNa = 2 * self.get_comp_factor(M)
        self.CNa = self.distribute(CNa, self.lat_area)

    @staticmethod
    def _get_vk_profile(x, L, R) -> np.ndarray:

        theta = np.arccos(1 - 2 * x / L)
        r = (R / np.sqrt(np.pi)) * np.sqrt(
            theta - 0.5 * np.sin(2 * theta)
        )

        return r
    
    #@staticmethod
    #def _get_power_series(x, L, R, n) -> np.ndarray: