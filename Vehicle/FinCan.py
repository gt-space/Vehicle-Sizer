import numpy as np
import matproplib as mp
from Vehicle.Section import Section

class FinCan(Section):

    def __init__(self, cfg: dict):

        super().__init__(cfg)
        self.length = cfg["engine"]["length"]
        self.n = int(np.ceil(self.length / self.dx))

    def get_mass(self):

        engine_mass = self.cfg["engine"]["mass"]
        motor_mass = 2
        fin_mass = self._get_fin_mass()

        const_mass = np.full(self.n, (fin_mass + engine_mass + motor_mass) / self.n)
        boattail_mass = self._get_boattail_mass_distribution()
        self.mass = const_mass + boattail_mass
    
    def _get_fin_mass(self):

        A = self.cfg["fin_can"]["fin_area"]
        t = self.cfg["fin_can"]["fin_thickness"]
        n = self.cfg["fin_can"]["fin_count"]

        V = A * t * n
        mat = mp.db.get_material(self.cfg["fin_can"]["material"])
        rho = mat.get("density")
        m = rho * V

        return m

    def _get_boattail_mass(self):

        r_s_o = self.cfg["vehicle"]["OMLD"] * 0.5
        t = self.cfg["fin_can"]["boattail_wall_thickness"]
        r_s_i = r_s_o - t

        Ae = 0.025
        r_f_i = np.sqrt(Ae / np.pi)
        r_f_o = r_f_i + t

        V_o = (1/3) * np.pi * self.length * (r_s_o**2 + r_s_o * r_f_o + r_f_o**2)
        V_i = (1/3) * np.pi * self.length * (r_s_i**2 + r_s_i * r_f_i + r_f_i**2)

        V = V_o - V_i
        mat = mp.db.get_material(self.cfg["fin_can"]["material"])
        rho = mat.get("density")
        m = rho * V

        return m
    
    def _get_boattail_mass_distribution(self):

        r_s_o = self.cfg["vehicle"]["OMLD"] * 0.5
        t = self.cfg["fin_can"]["boattail_wall_thickness"]
        r_s_i = r_s_o - t

        Ae = 0.025
        r_f_i = np.sqrt(Ae / np.pi)
        r_f_o = r_f_i + t

        mat = mp.db.get_material(self.cfg["fin_can"]["material"])
        rho = mat.get("density")

        x_local = np.arange(self.n) * self.dx

        r_o = r_s_o + (x_local / self.length) * (r_f_o - r_s_o)
        r_i = r_o - t

        dV = np.pi * (r_o**2 - r_i**2) * self.dx

        return rho * dV

    def get_EI(self):

        r_s_o = self.cfg["vehicle"]["OMLD"] * 0.5
        t = self.cfg["fin_can"]["boattail_wall_thickness"]
        r_s_i = r_s_o - t

        Ae = 0.025
        r_f_i = np.sqrt(Ae / np.pi)
        r_f_o = r_f_i + t

        x_local = np.arange(self.n) * self.dx

        r_o = r_s_o + (x_local / self.length) * (r_f_o - r_s_o)
        r_i = r_o - t

        I = (np.pi / 4.0) * (r_o**4 - r_i**4)

        mat = mp.db.get_material(self.cfg["fin_can"]["material"])
        T = 300.0
        E = mat.get("elastic_modulus_0deg", T)

        self.EI = E * I

    def get_area(self):

        self.lat_area = np.full(self.n, 1)
        self.surf_area = self.lat_area