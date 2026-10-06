import numpy as np
from .Section import Section
from ..COPV import COPV
from ..Material import MaterialProperties
from ..utils import distribute as dist
from ..utils import geometry as geo



from ..tank_geometry import PressTankGeometry

class PressTank(Section):

    def __init__(self, cfg: dict, copv: COPV, tank_id: str):

        super().__init__(cfg)
        self.tank_id = tank_id
        self.vessel = copv
        self.copv = copv # compatibility alias for existing geometry consumers
        self.length = self.copv.length
        self.set_grid()
        self.wall_thickness = copv.wall_thickness
        definition = cfg["tanks"][tank_id]
        self.wall_material = definition.get("material", cfg["nosecone"]["material"])
        self.emissivity = 0.85

    def get_fluid_geometry(self) -> PressTankGeometry:
        """Export immutable internal geometry for the fluid network."""

        return PressTankGeometry(
            volume=self.copv.volume,
            length=self.copv.length,
            inner_diameter=self.copv.inner_diameter,
            cylinder_length=self.copv.cylinder_length,
            ellipse_ratio=self.copv.ellipse_ratio,
            internal_area=self.copv.internal_area,
            resolution=self.n,
        )

    def get_mass(self):
        self.shell_mass = dist.uniform(self.copv.mass, self.n)
        mass = self.copv.mass
        self.dry_mass = dist.uniform(mass, self.n)
        self.mass = self.dry_mass.copy()

    def set_fluid_mass(self, axial_mass: np.ndarray) -> None:
        """Add a network-supplied gas vector to this tank's dry mass."""

        axial_mass = np.asarray(axial_mass, dtype=float)
        if axial_mass.shape != self.dry_mass.shape:
            raise ValueError(
                f"Tank '{self.tank_id}' requires {self.n} axial mass values"
            )
        if np.any(axial_mass < 0.0):
            raise ValueError(f"Tank '{self.tank_id}' fluid mass cannot be negative")
        self.mass = self.dry_mass + axial_mass
        self.get_MOI()

    def get_EI(self):
        self.EI = np.zeros(self.n)

    def get_area(self):
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        self.lat_area = dist.uniform(geo.cylinder_lateral_area(r, self.length), self.n)
        self.surf_area = dist.uniform(geo.cylinder_surface_area(r, self.length), self.n)

    def get_MOI(self):
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        self.cg = (np.sum(self.mass * self.station) / np.sum(self.mass)
                   if np.sum(self.mass) > 0 else np.mean(self.station))
        self.Ixx = np.sum(self.mass * r**2)
        self.Iyy = np.sum(self.mass * (self.station - self.cg)**2)


    def get_thermal_oml_area(self) -> np.ndarray:
        return self.surf_area.copy()

    def get_thermal_shell_mass(self) -> np.ndarray:
        return self.shell_mass.copy()

    def get_thermal_internal_area(self) -> np.ndarray:
        return np.full(self.n, self.copv.internal_area / self.n)
