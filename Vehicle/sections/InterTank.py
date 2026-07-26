import numpy as np
from scipy.optimize import root
import matproplib as mp
from .Section import Section
from ..utils import distribute as dist
from ..utils import aero
from ..utils import geometry as geo
from ..utils import heating

class InterTank(Section):

    def __init__(self, cfg: dict, length: float, ax_load: float, bending_moment: float):

        super().__init__(cfg)
        self.length = length
        self.n = int(np.ceil(self.length / self.dx))
        self.ax_load = ax_load
        self.bending_moment = bending_moment

    def get_mass(self):
        feed_system_mass = self.cfg["inter_tank"]["feed_system_mass"]
        avi_mass = self.cfg["inter_tank"]["avi_mass"]
        stringer_mass, self.stringer_thickness = self._get_stringer_mass()
        mass = self._get_clamshell_mass() + stringer_mass + feed_system_mass + avi_mass
        self.mass = dist.uniform(mass, self.n)

    def _get_clamshell_mass(self) -> float:
        t = self.cfg["inter_tank"]["clamshell_wall_thickness"]
        r_o = self.cfg["vehicle"]["OMLD"] * 0.5
        r_i = r_o - t
        rho = mp.db.get_material(self.cfg["inter_tank"]["clamshell_material"]).get("density")
        return geo.annulus_volume(r_o, r_i, self.length) * rho

    def _get_stringer_mass(self) -> float:
        mat = mp.db.get_material(self.cfg["inter_tank"]["stringer_material"])
        sigma = mat.get("yield_strength", 350.0)
        E = mat.get("elastic_modulus", 350.0)
        a = self._get_stringer_thickness(self.ax_load, self.bending_moment, sigma, E)
        m = a**2 * self.length * 4 * mat.get("density")
        return m, a

    def _get_stringer_thickness(self, P: float, M: float, sigma: float, E: float) -> float:

        r = self.cfg["vehicle"]["OMLD"] * 0.5
        FOS = 1.5

        def equations(x):
            a, I = x
            if a <= 0 or I <= 0:
                return [1e9, 1e9]
            eq1 = (M * np.sqrt(r**2 - 0.25 * a**2)) / (I + (P / (4 * a**2))) - sigma
            eq2 = (2 * a**2) * ((a**2) / 3 + r**2 - r * a) - I
            return [eq1, eq2]

        a_guess = 0.0001
        I_guess = (2 * a_guess**2) * ((a_guess**2) / 3 + r**2 - r * a_guess)
        sol = root(equations, [a_guess, I_guess])
        return sol.x[0] * FOS

    def get_EI(self):
        EI = self._get_clamshell_EI() + self._get_stringer_EI(self.stringer_thickness)
        self.EI = dist.uniform_full(EI, self.n)

    def _get_clamshell_EI(self) -> float:
        t = self.cfg["inter_tank"]["clamshell_wall_thickness"]
        r_o = self.cfg["vehicle"]["OMLD"] * 0.5
        r_i = r_o - t
        E = mp.db.get_material(self.cfg["inter_tank"]["clamshell_material"]).get("elastic_modulus_0deg", 300.0)
        return E * geo.annulus_second_moment(r_o, r_i)

    def _get_stringer_EI(self, a: float) -> float:
        t = self.cfg["inter_tank"]["clamshell_wall_thickness"]
        r_o = self.cfg["vehicle"]["OMLD"] * 0.5
        r = r_o - t - a * 0.5
        I = 4 * (a**4 / 12 + a**2 * r**2)
        E = mp.db.get_material(self.cfg["inter_tank"]["stringer_material"]).get("elastic_modulus", 300.0)
        return E * I

    def get_area(self):
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        self.lat_area = dist.uniform(geo.cylinder_lateral_area(r, self.length), self.n)
        self.surf_area = dist.uniform(geo.cylinder_surface_area(r, self.length), self.n)

    def get_MOI(self):
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        self.cg = np.sum(self.mass * self.station) / np.sum(self.mass)
        self.Ixx = np.sum(self.mass * r**2)
        self.Iyy = np.sum(self.mass * (self.station - self.cg)**2)

    def get_CNa(self, M: float, alpha: float):
        A_plan = self.cfg["vehicle"]["OMLD"] * self.length
        self.CNa = dist.weighted(aero.body_CNa(M, alpha, A_plan, self.ref_area), self.lat_area)

    def get_heat_flux(self, ):
        self.heat_flux = heating.get_body_heating()