import numpy as np
import matproplib as mp
from .Section import Section
from .COPV import COPV

class PressTank(Section):

    def __init__(self, cfg: dict, copv: COPV):
        
        super().__init__(cfg)
        self.copv = copv
        self.length = self.copv.length
        self.n = int(np.ceil(self.length / self.dx))

    def get_mass(self):

        mass = self._get_mount_mass() + self._get_airframe_mass() + self.copv.mass
        self.mass = np.full(self.n, mass / self.n)

    def _get_mount_mass(self) -> float:

        mat = mp.db.get_material(self.cfg["press_tank"]["mount_material"])
        T = 400.0
        sigma = mat.get("yield_strength")
        FOS = 1.4

        a = 9.81 * 10.0
        P = self.copv.mass * a

        h = self.cfg["press_tank"]["mount_thickness"]
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        L = r - self.cfg["press_tank"]["airframe_wall_thickness"]
        w = FOS * (3 * P * L) / (2 * sigma * h**2)

        n = 4
        V = w * L * h * n
        rho = mat.get("density")
        m = V * rho

        return m

    def _get_airframe_mass(self) -> float:

        t = self.cfg["press_tank"]["airframe_wall_thickness"]
        r_o = self.cfg["vehicle"]["OMLD"] * 0.5
        A_o = np.pi * r_o**2

        r_i = r_o - t
        A_i = np.pi * r_i**2

        A = A_o - A_i
        V = A * self.length
        mat = mp.db.get_material(self.cfg["press_tank"]["airframe_material"])
        rho = mat.get("density")
        m = V * rho

        return m
    
    def get_EI(self):

        r_o = self.cfg["vehicle"]["OMLD"] * 0.5

        t = self.cfg["press_tank"]["airframe_wall_thickness"]
        r_i = r_o - t

        I = np.pi * 0.25 * (r_o**4 - r_i**4)
        mat = mp.db.get_material(self.cfg["press_tank"]["airframe_material"])
        T = 300.0
        E = mat.get("elastic_modulus_0deg", T)
        EI = E * I

        self.EI = np.full(self.n, EI)
    
    def get_area(self):

        D = self.cfg["vehicle"]["OMLD"]
        self.lat_area = np.full(self.n, D * self.dx)
        self.surf_area = self.lat_area * np.pi

    def get_CNa(self, M: float, alpha: float):

        K = 1.1
        P = self.get_comp_factor(M)
        A_plan = self.length * self.cfg["vehicle"]["OMLD"]
        CNa = K * P * (A_plan / self.ref_area) * (np.sin(alpha)**2 / alpha)

        self.CNa = self.distribute(CNa, self.lat_area)