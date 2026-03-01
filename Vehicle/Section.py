from abc import ABC, abstractmethod

class Section(ABC):

    def __init__(self, cfg: dict):

        self.cfg = cfg
        
        self.dx = cfg["vehicle"]["dx"]
        self.length = None
        self.n = None

        self.x = None
        self.x_start = None
        self.x_end = None
        
        self.mass = None
        self.EI = None
        self.lat_area = None
        
        self.radius = None
        self.front_area = None

    def build(self):
        self.get_mass()
        self.get_EI()
        self.get_lat_area()

    @abstractmethod
    def get_mass(self):
        pass

    @abstractmethod
    def get_EI(self):
        pass

    @abstractmethod
    def get_lat_area(self):
        pass