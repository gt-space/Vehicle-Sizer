from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class SectionInputs:
    axial_load: float
    bending_moment: float
    temp: float

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
        self.cg: float = None

        self.lat_area: np.ndarray = None
        self.surf_area: np.ndarray = None

        self.ref_area: float = 1
        self.CNa: np.ndarray = None

    def build(self):
        self.get_mass()
        self.get_EI()
        self.get_area()
        self.get_MOI()

    @abstractmethod
    def get_mass(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_EI(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_area(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_MOI(self):
        pass

    @abstractmethod
    def get_CNa(self, M: float, alpha: float) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_heat_flux() -> np.ndarray:
        pass
    
    @abstractmethod
    def get_temp() -> np.ndarray:
        pass