from abc import ABC, abstractmethod
import numpy as np

class Section(ABC):

    def __init__(self, cfg: dict):

        self.cfg: dict = cfg
        
        self.dx: float = cfg["vehicle"]["dx"]
        self.length: float = None
        self.n: int = None

        self.station: np.ndarray = None
        self.start_station: float = None
        self.end_station: float = None

        self.ax_load: float = None
        self.bending_moment: float = None

        self.mass: np.ndarray = None
        self.EI: np.ndarray = None

        self.lat_area: np.ndarray = None
        self.surf_area: np.ndarray = None

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