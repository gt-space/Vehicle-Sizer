import numpy as np
from .Section import Section
from ..utils import distribute as dist
from ..utils import geometry as geo

class InterTank(Section):

    def __init__(self, cfg: dict, length: float, stiffness_EI: float,
                 *, stringer_unit_mass: float):

        super().__init__(cfg)
        self.stringer_unit_mass = float(stringer_unit_mass)
        if not np.isfinite(self.stringer_unit_mass) or self.stringer_unit_mass < 0:
            raise ValueError("stringer_unit_mass must be finite and nonnegative")
        self.length = length
        self.set_grid()
        self.stiffness_input = float(stiffness_EI)
        if not np.isfinite(self.stiffness_input) or self.stiffness_input <= 0:
            raise ValueError("stiffness_EI must be finite and positive (N m^2)")
        self.stringer_count = int(cfg["inter_tank"]["stringer_count"])
        if self.stringer_count < 3:
            raise ValueError("Intertank requires at least three stringers")
        self.wall_thickness = 0.0
        self.wall_material = cfg["inter_tank"]["stringer_material"]
        self.emissivity = 0.85

    def get_mass(self):
        self.shell_mass = np.zeros(self.n)
        # Unit mass is per stringer; count comes from inter_tank.stringer_count.
        mass = self.stringer_unit_mass * self.length * self.stringer_count
        self.mass = dist.uniform(mass, self.n)

    def get_EI(self):
        self.EI = np.full(self.n, self.stiffness_input)

    def get_area(self):
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        self.lat_area = dist.uniform(geo.cylinder_lateral_area(r, self.length), self.n)
        self.surf_area = dist.uniform(geo.cylinder_surface_area(r, self.length), self.n)

    def get_MOI(self):
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        self.cg = (np.sum(self.mass * self.station) / np.sum(self.mass)
                   if np.sum(self.mass) > 0 else np.mean(self.station))
        self.Ixx = np.sum(self.mass * r**2)
        self.Iyy = np.sum(self.mass * (self.station - self.cg)**2)


    def get_thermal_oml_area(self) -> np.ndarray:
        return self.surf_area.copy()

    def get_thermal_shell_mass(self) -> np.ndarray:
        return self.shell_mass.copy()
