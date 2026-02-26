from abc import ABC, abstractmethod

class Section(ABC):

    def __init__(self, cfg: dict):

        self.cfg = cfg
        
        self.length = None

        self.station = None

        self.mass = None
        self.EI = None
        
        self.radius = None
        self.lat_area = None
        self.front_area = None

    @abstractmethod
    def get_mass(self):
        pass

    @abstractmethod
    def get_EI(self):
        pass