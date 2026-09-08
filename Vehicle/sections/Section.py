from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import matproplib as mp

SIGMA = 5.670374419e-8                          # Stefan-Boltzmann, W/m^2-K^4

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

        self.Tw: np.ndarray = None
        self.heat_flux: np.ndarray = None
        self.wall_thickness: float = None
        self.wall_material: str = None
        self.emissivity: float = None

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
    
    def init_thermal(self, T0: float):
        self.Tw = np.full(self.n, T0)

    @abstractmethod
    def get_heat_flux(self, atm, theta: float):
        pass

    def get_temp(self, atm, dt: float):
        # transient lumped-mass wall: convection in, grey-body re-radiation out
        rho_w = mp.db.get_material(self.wall_material).get("density")
        c_w = mp.db.get_material(self.wall_material).get("specific_heat")
        areal_cap = rho_w * c_w * self.wall_thickness
        q_rad = self.emissivity * SIGMA * (self.Tw**4 - atm.T**4)
        self.Tw = self.Tw + dt * (self.heat_flux - q_rad) / areal_cap