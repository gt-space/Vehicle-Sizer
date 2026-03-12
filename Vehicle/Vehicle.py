import numpy as np

class Vehicle:

    def __init__(self, cfg: dict, sections: list):

        self.cfg: dict = cfg
        self.sections: list = sections

        self.station: np.ndarray = None
        self.mass: np.ndarray = None
        self.EI: np.ndarray = None
        self.lat_area: np.ndarray = None
        self.surf_area: np.ndarray = None
        self.CNa: np.ndarray = None

        self.length: float = None
        self.total_mass: float = None
        self.cg: float = None
        self.Ixx: float = None
        self.Iyy: float = None

    def build(self):

        self._stack_sections()
        self._assemble_vectors()
        self.get_mass_properties()

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
        self.surf_area = np.concatenate([sec.surf_area for sec in self.sections])

    def get_mass_properties(self):
        
        self.total_mass = np.sum(self.mass)
        self.cg = np.sum(self.mass * self.station) / self.total_mass

    def get_CNa(self, M: float, alpha: float):

        for sec in self.sections:
            sec.get_CNa(M, alpha)

        self.CNa = np.concatenate([sec.CNa for sec in self.sections])

    # def update(self):