import numpy as np
import matproplib as mp
import Vehicle.Section as Section
from abc import ABC, abstractmethod

class PropTank(Section, ABC):

    def __init__(self, cfg: dict):
        
        super().__init__(cfg)

    def get_mass(self):

    def _get_dry_mass(self):

    @abstractmethod
    def _get_pressure(self):
        pass

    @abstractmethod
    def _get_volume(self):
        pass

    def _get_wall_thickness(self):

        mat = mp.db.get_material(self.cfg["prop_tank"]["material"])
        sigma = mat.get("yield_strength", T = 400.0)
        FOS = 1.4

        t = FOS * (self.pressure * self.diameter) / (2 * sigma)
        t = max(t, t_min)

        return t
    
    def _get_length(self):

        D = self.cfg["vehicle"]["OMLD"]

    def get_EI(self):

        r_o = self.cfg["vehicle"]["OMLD"] * 0.5

        t = self._get_wall_thickness()
        r_i = r_o - t

        I = np.pi * 0.25 * (r_o**4 - r_i**4)
        mat = mp.db.get_material(self.cfg["prop_tank"]["material"])
        T = 400.0
        E = mat.get("elastic_modulus")

        EI = E * I