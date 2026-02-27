import numpy as np
import matproplib as mp
from Vehicle.Section import Section

class FinCan(Section):

    def __init__(self, cfg: dict):

        super().__init__(cfg)
        self.n = int(np.ceil(self.length / self.dx))

    def get_mass(self):

        engine_mass = self.cfg["fin_can"]["engine_mass"]
        mass = self._get_fin_mass() + self._get_boattail_mass() + engine_mass

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
        r_s_i = r_s_o - self.cfg["fin_can"]["boattail_wall_thickness"]

    def get_EI(self):

    def get_lat_area(self):