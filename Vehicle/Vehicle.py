import numpy as np

class Vehicle:

    def __init__(self, cfg: dict, sections: list):

        self.cfg = cfg
        self.sections = sections

        self.station = None
        self.mass = None
        self.EI = None
        self.lat_area = None
        self.surf_area = None

        self.length = None
        self.total_mass = None
        self.cg = None
        self.Ixx = None
        self.Iyy = None

    def build(self):

        self._stack_sections()
        self._assemble_vectors()
        self._get_mass_properties()

    def _stack_sections(self):

        x_current = 0.0

        for sec in self.sections:

            sec.build()

            sec.start_station = x_current
            sec.end_station = x_current + sec.length

            sec.station = sec.start_station + np.arange(sec.n) * sec.dx

            x_current = sec.end_station

        self.length = x_current

    def _assemble_vectors(self):

        self.station = np.concatenate([sec.station for sec in self.sections])
        self.mass = np.concatenate([sec.mass for sec in self.sections])
        self.EI = np.concatenate([sec.EI for sec in self.sections])
        self.lat_area = np.concatenate([sec.lat_area for sec in self.sections])

    def _get_mass_properties(self):
        
        self.total_mass = np.sum(self.mass)
        self.cg = np.sum(self.mass * self.station) / self.total_mass

    # def update(self):