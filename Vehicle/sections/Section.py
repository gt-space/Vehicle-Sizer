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
        self.cg: float = None

        self.lat_area: np.ndarray = None
        self.surf_area: np.ndarray = None

        self.ref_area: float = 1
        self.CNa: np.ndarray = None

    def build(self):
        self.get_mass()
        self.apply_mass_inputs()
        self.build_stiffness()
        self.get_area()
        self.get_MOI()

    def build_stiffness(self):
        """Build local bending rigidity without depending on section name/order."""
        configured = getattr(self, "stiffness_input", None)
        if configured is None:
            self.get_EI()
        else:
            value = float(configured)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("stiffness_EI must be finite and positive (N m^2)")
            self.EI = np.full(self.n, value)
        self.EI = np.asarray(self.EI, dtype=float)
        if self.EI.shape != (self.n,) or np.any(~np.isfinite(self.EI)) or np.any(self.EI < 0):
            raise ValueError("Section stiffness must have one finite nonnegative value per cell")

    def apply_mass_inputs(self):
        """Override calculated section dry mass, then add freely named hardware."""
        inputs = getattr(self, "mass_inputs", {})
        override = inputs.get("mass_override", inputs.get("mass"))
        manual = inputs.get("masses", {})
        if not isinstance(manual, dict):
            raise ValueError("masses must be a mapping of names to kg")
        values = np.asarray(list(manual.values()), dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("manual masses must be finite and nonnegative")
        if override is not None:
            override = float(override)
            if not np.isfinite(override) or override <= 0:
                raise ValueError("mass override must be finite and positive")
            calculated_total = float(np.sum(self.mass))
            total = override
            self.mass = np.full(self.n, total / self.n)
            if hasattr(self, "shell_mass"):
                self.shell_mass = self.shell_mass * (override / calculated_total) if calculated_total > 0 else np.zeros(self.n)
        else:
            self.mass = self.mass + values.sum() / self.n
        if hasattr(self, "dry_mass"):
            self.dry_mass = self.mass.copy()

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
