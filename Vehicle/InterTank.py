import numpy as np
from scipy.optimize import root
import matproplib as mp
from .Section import Section

class InterTank(Section):

    def __init__(self, cfg: dict, L: float, P: float, M: float):

        super().__init__(cfg)
        self.length = L
        self.n = int(np.ceil(self.length / self.dx))
        self.P = P
        self.M = M

    def get_mass(self):

        feed_system_mass = self.cfg["inter_tank"]["feed_system_mass"]
        avi_mass = self.cfg["inter_tank"]["avi_mass"]
        stringer_mass, self.stringer_thickness = self._get_stringer_mass()
        
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

    def _get_stringer_mass(self):
        
        T = 350.0
        mat = mp.db.get_material(self.cfg["inter_tank"]["stringer_material"])
        sigma = mat.get("yield_strength", T)
        E = mat.get("elastic_modulus", T)

        a = self._get_stringer_thickness(self.P, self.M, sigma, E)

        n = 4
        A = a**2
        V = A * self.length * n

        rho = mat.get("density")
        m = V * rho

        return m, a
    
    def _get_stringer_thickness(self, P, M, sigma, E):

        r = self.cfg["vehicle"]["OMLD"] * 0.5

        def equations(x):

            a, I = x

            if a <= 0 or I <= 0:
                return [1e9, 1e9]

            eq1 = (M * np.sqrt(r**2 - 0.25 * a**2)) / (I + (P / (4 * a**2))) - sigma
            eq2 = (2 * a**2) * ((a**2) / 3 + r**2 - r * a) - I

            return [eq1, eq2]
        
        a_guess = 0.005
        I_guess = (2 * a_guess**2) * ((a_guess**2) / 3 + r**2 - r * a_guess)

        sol = root(equations, [a_guess, I_guess])
        #a = sol.x[0]

        n = 1
        FOS = 1.5
        Le = self.length * n

        a_buckling = ((12 * Le**2 * P) / (np.pi**2 * E))**0.25 * FOS
        a_yielding = np.sqrt(P / sigma) * FOS
        a = max(a_buckling, a_yielding)

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