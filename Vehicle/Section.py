from abc import ABC, abstractmethod

class Section(ABC):

    def __init__(self, cfg: dict):

        self.cfg = cfg
        
        self.dx = cfg["vehicle"]["dx"]
        self.length = None
        self.n = None

        self.station = None
        self.start_station = None
        self.end_station = None

        self.ax_load = None
        self.bending_moment = None

        self.mass = None
        self.EI = None

        self.lat_area = None
        self.surf_area = None
        self.radius = None

    def build(self):
        
        self.get_mass()
        self.get_EI()
        self.get_area()

    @abstractmethod
    def get_mass(self):
        pass

    @abstractmethod
    def get_EI(self):
        pass

    @abstractmethod
    def get_area(self):
        pass