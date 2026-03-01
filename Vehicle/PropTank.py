import numpy as np
import matproplib as mp
from Vehicle.Section import Section
from abc import ABC, abstractmethod

class PropTank(Section):

    def __init__(self, cfg: dict, tank_type: str):
        
        super().__init__(cfg)
        self.tank_cfg = cfg[tank_type]

    def get_mass(self):

    def _get_dry_mass(self):

    def _get_pressure(self):

    def _get_volume(self):

        

    def _get_wall_thickness(self):

        mat = mp.db.get_material(self.tank_cfg["material"])
        T = 400.0
        sigma = mat.get("yield_strength", T)
        FOS = 1.4

        t = FOS * (self.pressure * self.diameter) / (2 * sigma)
        t_min = 1/16 * 0.0254
        t = max(t, t_min)

        self.wall_thickness = t
    
    def _get_length(self):

        D = self.cfg["vehicle"]["OMLD"]
        D_pass = self.tank_cfg["passthrough_diameter"]
        V_end = (np.pi / (12 * self.tank_cfg["ellipse_ratio"])) * (D - (2 * self.wall_thickness))^3

        L_cyl = 
        L = L_cyl + (D / self.tank_cfg["ellipse_ratio"])

    def get_EI(self):

        r_o = self.cfg["vehicle"]["OMLD"] * 0.5

        t = self._get_wall_thickness()
        r_i = r_o - t

        I = np.pi * 0.25 * (r_o**4 - r_i**4)
        mat = mp.db.get_material(self.tank_cfg["material"])
        T = 400.0
        E = mat.get("elastic_modulus", T)

        EI = E * I
        self.EI = np.full(self.n, EI)

    def get_lat_area(self):

        D = self.cfg["vehicle"]["OMLD"]
        self.lat_area = np.full(self.n, D * self.dx)