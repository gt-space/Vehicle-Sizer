import numpy as np
from scipy.optimize import root_scalar
import matproplib as mp
import Vehicle.Section as Section

class AviBay(Section):

    def __init__(self, cfg: dict):
        
        super().__init__(cfg)

    def get_mass(self):
        
        mass = self._get_bulkhead_mass() + self.cfg["avi_bay"]["avi_mass"]

    def _get_bulkhead_mass(self):
        
        a = 9.81 * 10.0
        P = self.cfg["avi_bay"]["avi_mass"] * a

        mat = mp.get_material(self.cfg["avi_bay"]["bulkhead_material"])
        nu = mat.get("poisson_ratio")
        T = 350.0
        sigma = mat.get("yield_strength_90deg", T)

        r = (self.cfg["vehicle"]["OMLD"] * 0.5) - self.cfg["avi_bay"]["clamshell_thickness"]

        t = self._get_bulkhead_thickness(P, r, nu, sigma)
        A = np.pi * r**2
        V = A * t

        rho = mat.get("density")
        m = V * rho

        return m

    def _get_bulkhead_thickness(self, P, r, nu, sigma):

        FOS = 1.4

        def f(t):
            return (P / t**2) * (1 + nu) * (0.485 * np.log(r / t) + 0.52) - sigma
        
        t_bounds = (1e-3, 0.1)
        sol = root_scalar(f, bracket=t_bounds, method='brentq')

        t = sol.root * FOS

        return t

    def get_EI(self):

    def get_lat_area(self):