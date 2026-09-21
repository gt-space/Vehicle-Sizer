from abc import ABC, abstractmethod
import numpy as np

# Legacy analytical aero / unused input container (inactive).
# @dataclass
# class SectionInputs:
#     axial_load: float
#     bending_moment: float
#     temp: float

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

    def set_grid(self):
        """Cells exactly cover the section; stations denote cell centers [m]."""
        if not np.isfinite(self.length) or self.length <= 0 or not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError("Section length and requested cell spacing must be finite and positive")
        self.n = max(2, int(np.ceil(self.length / self.dx)))
        self.dx = self.length / self.n
        self.local_edges = np.linspace(0.0, self.length, self.n + 1)
        self.local_centers = 0.5 * (self.local_edges[:-1] + self.local_edges[1:])

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

# Legacy analytical aero / unused input container (inactive).
#     @abstractmethod
#     def get_CNa(self, M: float, alpha: float) -> np.ndarray:
#         pass

    @abstractmethod
    def get_thermal_oml_area(self) -> np.ndarray:
        """Return external heated area [m^2] for each axial cell."""
        pass

    @abstractmethod
    def get_thermal_shell_mass(self) -> np.ndarray:
        """Return thermally active shell mass [kg] for each axial cell."""
        pass

    def get_thermal_internal_area(self) -> np.ndarray:
        """Return internal heat-transfer area [m^2] for each axial cell."""

        return self.get_thermal_oml_area()
