import numpy as np
from .Section import Section

class InterTank(Section):

    def __init__(self, cfg: dict, length: float, stiffness_EI: float):

        super().__init__(cfg)
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
        # All intertank hardware mass is supplied through the section mass inputs.
        self.mass = np.zeros(self.n)

    def get_EI(self):
        self.EI = np.full(self.n, self.stiffness_input)
