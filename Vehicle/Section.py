from abc import ABC, abstractmethod

class Section(ABC):

    def __init__(self, cfg: dict):

        self.cfg = cfg
        
        self.length = None
        self.dx = cfg["vehicle"]["dx"]
        self.n = None

        self.station = None

        self.mass = None
        self.EI = None
        self.lat_area = None
        
        self.radius = None
        self.front_area = None

    def build(self):
        self.mass = self.get_mass()
        self.EI = self.get_EI()
        self.lat_area = self.get_lat_area()

    @abstractmethod
    def get_mass(self):
        pass

    @abstractmethod
    def get_EI(self):
        pass

    @abstractmethod
    def get_lat_area(self):
        pass