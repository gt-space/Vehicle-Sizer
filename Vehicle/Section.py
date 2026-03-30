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

        self.Ixx: float = None
        self.Iyy: float = None

        self.lat_area: np.ndarray = None
        self.surf_area: np.ndarray = None

        self.ref_area: float = 1.0
        self.CNa: np.ndarray = None

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

    @abstractmethod
    def get_CNa(self, M: float, alpha: float):
        pass

    @staticmethod
    def distribute(C: float, vec: np.ndarray) -> np.ndarray:
        return C * (vec / np.sum(vec))

    @staticmethod
    def get_comp_factor(M: float) -> float:

        if M <= 0.8:
            return 1 / np.sqrt(1 - M**2)

        elif M < np.sqrt(34 / 5):
            return 1 / np.sqrt(1 - 0.8**2)

        else:
            return 1 / np.sqrt(M**2 - 1)