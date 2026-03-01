import numpy as np

class Vehicle:

    def __init__(self, cfg: dict, sections: list):

        self.cfg = cfg
        self.sections = sections

        self.x = None
        self.mass = None
        self.EI = None
        self.lat_area = None

        self.length = None
        self.total_mass = None
        self.cg = None
        self.Ixx = None
        self.Iyy = None

    def build(self):

        self._build_sections()
        self._stack_sections()
        self._assemble_vectors()

    def _build_sections(self):
        for sec in self.sections:
            sec.build()

    def _stack_sections(self):

        x_current = 0.0

        for sec in self.sections:

            sec.x_start = x_current
            sec.x_end = x_current + sec.length

            sec.x = sec.x_start + np.arange(sec.n) * sec.dx

            x_current = sec.x_end

        self.length = x_current

    def _assemble_vectors(self):

        self.x = np.concatenate([sec.x for sec in self.sections])
        self.mass = np.concatenate([sec.mass for sec in self.sections])
        self.EI = np.concatenate([sec.EI for sec in self.sections])
        self.lat_area = np.concatenate([sec.lat_area for sec in self.sections])

    # def update(self):