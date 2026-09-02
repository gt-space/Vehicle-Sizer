import numpy as np
import matproplib as mp
from .Section import Section
from ..COPV import COPV
from ..utils import distribute as dist
from ..utils import aero
from ..utils import geometry as geo
from ..utils import heating
from dataclasses import dataclass

@dataclass
class PressTankInputs:
    copv: COPV
    mount_material: str
    mount_thickness: float
    airframe_material: str


@dataclass(frozen=True)
class PressTankGeometry:
    """Immutable COPV geometry used by the gas-volume node."""

    volume: float
    length: float
    diameter: float
    internal_area: float
    resolution: int = 1

    def __post_init__(self):
        if any(
            value <= 0.0
            for value in (self.volume, self.length, self.diameter, self.internal_area)
        ):
            raise ValueError("Pressure tank geometry values must be positive")
        if self.resolution < 1:
            raise ValueError("Pressure tank geometry resolution must be positive")

    def axial_mass(self, mass: float) -> np.ndarray:
        """Return the local fore-to-aft gas mass vector."""

        if mass < 0.0:
            raise ValueError("Pressure-tank gas mass cannot be negative")
        return np.full(self.resolution, mass / self.resolution)

class PressTank(Section):

    def __init__(self, cfg: dict, copv: COPV, tank_id: str):

        super().__init__(cfg)
        self.tank_id = tank_id
        self.copv = copv
        self.length = self.copv.length
        self.n = int(np.ceil(self.length / self.dx))

    def get_fluid_geometry(self) -> PressTankGeometry:
        """Export immutable internal geometry for the fluid network."""

        return PressTankGeometry(
            volume=self.copv.volume,
            length=self.copv.length,
            diameter=self.copv.diameter,
            internal_area=self.copv.internal_area,
            resolution=self.n,
        )

    def get_mass(self):
        mass = self._get_mount_mass() + self._get_airframe_mass() + self.copv.mass
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

    def _get_mount_mass(self) -> float:
        mat = mp.db.get_material(self.cfg["press_tank"]["mount_material"])
        sigma = mat.get("yield_strength", 400.0)
        P = self.copv.mass * 9.81 * 10.0
        h = self.cfg["press_tank"]["mount_thickness"]
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        L = r - self.cfg["press_tank"]["airframe_wall_thickness"]
        w = 1.4 * (3 * P * L) / (2 * sigma * h**2)
        return w * L * h * 4 * mat.get("density")

    def _get_airframe_mass(self) -> float:
        t = self.cfg["press_tank"]["airframe_wall_thickness"]
        r_o = self.cfg["vehicle"]["OMLD"] * 0.5
        r_i = r_o - t
        rho = mp.db.get_material(self.cfg["press_tank"]["airframe_material"]).get("density")
        return geo.annulus_volume(r_o, r_i, self.length) * rho

    def get_EI(self):
        t = self.cfg["press_tank"]["airframe_wall_thickness"]
        r_o = self.cfg["vehicle"]["OMLD"] * 0.5
        r_i = r_o - t
        E = mp.db.get_material(self.cfg["press_tank"]["airframe_material"]).get("elastic_modulus_0deg", 300.0)
        self.EI = dist.uniform_full(E * geo.annulus_second_moment(r_o, r_i), self.n)

    def get_area(self):
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        self.lat_area = dist.uniform(geo.cylinder_lateral_area(r, self.length), self.n)
        self.surf_area = dist.uniform(geo.cylinder_surface_area(r, self.length), self.n)

    def get_MOI(self):
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        self.cg = np.sum(self.mass * self.station) / np.sum(self.mass)
        self.Ixx = np.sum(self.mass * r**2)
        self.Iyy = np.sum(self.mass * (self.station - self.cg)**2)

    def get_CNa(self, M: float, alpha: float):
        A_plan = self.cfg["vehicle"]["OMLD"] * self.length
        self.CNa = dist.weighted(aero.body_CNa(M, alpha, A_plan, self.ref_area), self.lat_area)

    def get_heat_flux(self, ):
        self.heat_flux = heating.get_body_heating()
