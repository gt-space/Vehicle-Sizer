import numpy as np
import matproplib as mp
from .Section import Section
from ..Engine import Engine
from ..utils import distribute as dist
from ..utils import aero
from ..utils import geometry as geo

class FinCan(Section):

    def __init__(self, cfg: dict, engine: Engine):

        super().__init__(cfg)
        self.length = cfg["engine"]["length"]
        self.n = int(np.ceil(self.length / self.dx))
        self.engine = engine

    def get_mass(self):
        motor_mass = 2
        const_mass = dist.uniform(self._get_fin_mass() + self.engine.mass + motor_mass, self.n)
        self.mass = const_mass + self._get_boattail_mass_vector()

    def _get_fin_mass(self) -> float:
        A = self.cfg["fin_can"]["fin_area"]
        t = self.cfg["fin_can"]["fin_thickness"]
        n = self.cfg["fin_can"]["fin_count"]
        V = A * t * n
        mat = mp.db.get_material(self.cfg["fin_can"]["material"])
        rho = mat.get("density")
        m = rho * V
        return m

    def _get_boattail_mass(self) -> float:
        r_s_o = self.cfg["vehicle"]["OMLD"] * 0.5
        t = self.cfg["fin_can"]["boattail_wall_thickness"]
        r_s_i = r_s_o - t
        
        r_f_i = np.sqrt(self.engine.exit_area / np.pi)
        r_f_o = r_f_i + t

        V_o = (1/3) * np.pi * self.length * (r_s_o**2 + r_s_o * r_f_o + r_f_o**2)
        V_i = (1/3) * np.pi * self.length * (r_s_i**2 + r_s_i * r_f_i + r_f_i**2)

        V = V_o - V_i
        mat = mp.db.get_material(self.cfg["fin_can"]["material"])
        rho = mat.get("density")
        return rho * V

    def _get_boattail_mass_vector(self) -> np.ndarray:
        r_s_o = self.cfg["vehicle"]["OMLD"] * 0.5
        t = self.cfg["fin_can"]["boattail_wall_thickness"]
        
        r_f_i = np.sqrt(self.engine.exit_area / np.pi)
        r_f_o = r_f_i + t

        mat = mp.db.get_material(self.cfg["fin_can"]["material"])
        rho = mat.get("density")

        x_local = np.arange(self.n) * self.dx
        r_o = r_s_o + (x_local / self.length) * (r_f_o - r_s_o)
        r_i = r_o - t
        dV = geo.annulus_volume(r_o, r_i, self.dx)
        return rho * dV

    def get_EI(self):
        r_s_o = self.cfg["vehicle"]["OMLD"] * 0.5
        t = self.cfg["fin_can"]["boattail_wall_thickness"]

        Ae = 0.025
        r_f_i = np.sqrt(Ae / np.pi)
        r_f_o = r_f_i + t

        x_local = np.arange(self.n) * self.dx
        r_o = r_s_o + (x_local / self.length) * (r_f_o - r_s_o)
        r_i = r_o - t

        mat = mp.db.get_material(self.cfg["fin_can"]["material"])
        E = mat.get("elastic_modulus_0deg", 300.0)
        self.EI = E * geo.annulus_second_moment(r_o, r_i)

    def get_area(self):
        r_s_o = self.cfg["vehicle"]["OMLD"] * 0.5
        t = self.cfg["fin_can"]["boattail_wall_thickness"]

        r_f_i = np.sqrt(self.engine.exit_area / np.pi) + 0.01
        r_f_o = r_f_i + t

        x_local = np.arange(self.n) * self.dx
        r_o = r_s_o + (x_local / self.length) * (r_f_o - r_s_o)

        dr_dx = (r_f_o - r_s_o) / self.length
        ds = np.sqrt(self.dx**2 + (dr_dx * self.dx)**2)

        lat_body = 2.0 * r_o * self.dx
        surf_body = 2.0 * np.pi * r_o * ds

        total_fin_area = self.cfg["fin_can"]["fin_area"]

        n3 = int(self.n / 3)
        h_max = 2 * total_fin_area / self.length

        fin_heights = np.concatenate([
            np.linspace(0, h_max, n3),
            np.full(self.n - 2*n3, h_max),
            np.linspace(h_max, 0, n3)
        ])

        lat_fins = 2.0 * fin_heights * self.dx
        surf_fins = 2.0 * lat_fins

        self.lat_area = lat_body + lat_fins
        self.surf_area = surf_body + surf_fins
        self.lat_area_fins = lat_fins
        self.lat_area_body = lat_body

    def get_MOI(self):
        r_s_o = self.cfg["vehicle"]["OMLD"] * 0.5
        t = self.cfg["fin_can"]["boattail_wall_thickness"]
        r_f_o = np.sqrt(0.025 / np.pi) + t
        x_local = np.arange(self.n) * self.dx
        r = r_s_o + (x_local / self.length) * (r_f_o - r_s_o)
        self.cg = np.sum(self.mass * self.station) / np.sum(self.mass)
        self.Ixx = np.sum(self.mass * r**2)
        self.Iyy = np.sum(self.mass * (self.station - self.cg)**2)

    def get_CNa(self, M: float, alpha: float):
        Cr = self.length
        Ct = Cr / 3
        s = (3 * self.cfg["fin_can"]["fin_area"]) / (2 * Cr)
        N = self.cfg["fin_can"]["fin_count"]
        R_ref = self.cfg["vehicle"]["OMLD"] * 0.5

        fin_CNa = aero.fins_CNa(M, N, s, Cr, Ct, R_ref)
        tail_CNa = aero.taper_CNa(M, alpha, 0.75, self.ref_area)
        self.CNa = dist.weighted(fin_CNa, self.lat_area_fins) + dist.weighted(tail_CNa, self.lat_area_body)