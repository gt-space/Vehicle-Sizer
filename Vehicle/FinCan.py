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

        r_s_o = self.cfg["vehicle"]["OMLD"] * 0.5
        t = self.cfg["fin_can"]["boattail_wall_thickness"]

        Ae = 0.025
        r_f_i = np.sqrt(Ae / np.pi)
        r_f_o = r_f_i + t

        x_local = np.arange(self.n) * self.dx

        r_o = r_s_o + (x_local / self.length) * (r_f_o - r_s_o)

        dr_dx = (r_f_o - r_s_o) / self.length
        ds = np.sqrt(self.dx**2 + (dr_dx * self.dx)**2)

        lat_body = 2.0 * r_o * self.dx

        surf_body = 2.0 * np.pi * r_o * ds

        A_fin = self.cfg["fin_can"]["fin_area"]
        n_fin = self.cfg["fin_can"]["fin_count"]

        total_fin_area = A_fin

        dx = self.dx
        L = self.length

        n3 = int(self.n / 3)

        h_max = 2 * total_fin_area / L

        fin_heights = np.concatenate([
            np.linspace(0, h_max, n3),
            np.full(self.n - 2*n3, h_max),
            np.linspace(h_max, 0, n3)
        ])

        lat_fins = 2.0 * fin_heights * dx
        surf_fins = 2.0 * lat_fins

        self.lat_area = lat_body + lat_fins
        self.surf_area = surf_body + surf_fins

        self.lat_area_fins = lat_fins
        self.lat_area_body = lat_body