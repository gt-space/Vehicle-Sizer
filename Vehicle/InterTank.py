import numpy as np
from scipy.optimize import root
import matproplib as mp
import Vehicle.Section as Section

class InterTank(Section):

    def __init__(self, cfg: dict, L: float):

        super().__init__(cfg)
        self.length = L

    def get_mass(self):

        feed_system_mass = self.cfg["inter_tank"]["feed_system_mass"]
        avi_mass = self.cfg["inter_tank"]["avi_mass"]
        stringer_mass, self.stringer_thickness = self._get_stringer_mass(P, M)
        
        mass = self._get_clamshell_mass() + stringer_mass + feed_system_mass + avi_mass
        self.mass = np.full(self.n, mass / self.n)

    def _get_clamshell_mass(self):

        t = self.cfg["inter_tank"]["clamshell_wall_thickness"]
        r_o = self.cfg["vehicle"]["OMLD"] * 0.5
        A_o = np.pi * r_o**2

        r_i = r_o - t
        A_i = np.pi * r_i**2

        A = A_o - A_i

        mat = mp.db.get_material(self.cfg["inter_tank"]["clamshell_material"])
        rho = mat.get("density")
        m = A * self.length * rho

        return m

    def _get_stringer_mass(self, P, M):
        
        T = 400.0
        sigma = mp.db.get_material(self.cfg["inter_tank"]["stringer_material"]).get("yield_strength")

        a = self._get_stringer_thickness(P, M, sigma)

        n = 4
        A = a**2
        V = A * self.length * n

        mat = mp.db.get_material(self.cfg["inter_tank"]["stringer_material"])
        rho = mat.get("density")
        m = V * rho

        return m, a
    
    def _get_stringer_thickness(self, P, M, sigma):

        def equations(x):

            a, I = x

            if a <= 0 or I <= 0:
                return [1e9, 1e9]

            eq1 = (M * np.sqrt(r**2 - 0.25 * a**2)) / (I + (P / (4 * a**2))) - sigma
            eq2 = (2 * a**2) * ((a**2) / 3 + r**2 - r * a) - I

            return [eq1, eq2]
        
        a_guess = 0.0254
        I_guess = (2 * a_guess**2) * ((a_guess**2) / 3 + r**2 - r * a_guess)

        sol = root(equations, [a_guess, I_guess])
        a = sol.x[0]

        return a

    def get_EI(self):

        EI = self._get_clamshell_EI() + self._get_stringer_EI(self.stringer_thickness)
        self.EI = np.full(self.n, EI)

    def _get_clamshell_EI(self):

        r_o = self.cfg["vehicle"]["OMLD"] * 0.5

        t = self.cfg["inter_tank"]["clamshell_wall_thickness"]
        r_i = r_o - t

        I = np.pi * 0.25 * (r_o**4 - r_i**4)
        mat = mp.db.get_material(self.cfg["inter_tank"]["clamshell_material"])
        T = 400.0
        E = mat.get("elastic_modulus_0deg", T)
        EI = E * I

        return EI

    def _get_stringer_EI(self, a):

        r_o = self.cfg["vehicle"]["OMLD"] * 0.5
        t = self.cfg["inter_tank"]["clamshell_wall_thickness"]
        r_i = r_o - t
        r = r_i - (a * 0.5)

        n = 4
        I = n * (a**4 / 12 + a**2 * r**2)
        mat = mp.db.get_material(self.cfg["inter_tank"]["stringer_material"])
        T = 400.0
        E = mat.get("elastic_modulus", T)
        EI = E * I

        return EI
    
    def get_lat_area(self):

        D = self.cfg["vehicle"]["OMLD"]
        self.lat_area = np.full(self.n, D * self.dx)