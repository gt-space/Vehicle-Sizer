import numpy as np
from .Section import Section
from ..COPV import COPV
from ..Material import MaterialProperties
from ..utils import distribute as dist
# from ..utils import aero  # Legacy analytical aero disabled.
from ..utils import geometry as geo

# Legacy analytical aero / unused input container (inactive).
# @dataclass
# class PressTankInputs:
#     copv: COPV
#     mount_material: str
#     mount_thickness: float
#     airframe_material: str


from ..tank_geometry import PressTankGeometry

class PressTank(Section):

    def __init__(self, cfg: dict, copv: COPV, tank_id: str):

        super().__init__(cfg)
        self.tank_id = tank_id
        self.copv = copv
        self.length = self.copv.length
        self.set_grid()
        self.wall_thickness = cfg["press_tank"]["airframe_wall_thickness"]
        self.wall_material = cfg["press_tank"]["airframe_material"]
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
        self.shell_mass = dist.uniform(self._get_airframe_mass(), self.n)
        mass = self._get_mount_mass() + np.sum(self.shell_mass) + self.copv.mass
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
        mat = MaterialProperties.from_name(
            self.cfg["press_tank"]["mount_material"]
        )
        sigma = mat.require("yield_strength")
        P = self.copv.mass * 9.81 * 10.0
        h = self.cfg["press_tank"]["mount_thickness"]
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        L = r - self.cfg["press_tank"]["airframe_wall_thickness"]
        w = 1.4 * (3 * P * L) / (2 * sigma * h**2)
        return w * L * h * 4 * mat.density

    def _get_airframe_mass(self) -> float:
        t = self.cfg["press_tank"]["airframe_wall_thickness"]
        r_o = self.cfg["vehicle"]["OMLD"] * 0.5
        r_i = r_o - t
        rho = MaterialProperties.from_name(
            self.cfg["press_tank"]["airframe_material"]
        ).density
        return geo.annulus_volume(r_o, r_i, self.length) * rho

    def get_EI(self):
        t = self.cfg["press_tank"]["airframe_wall_thickness"]
        r_o = self.cfg["vehicle"]["OMLD"] * 0.5
        r_i = r_o - t
        E = MaterialProperties.from_name(
            self.cfg["press_tank"]["airframe_material"]
        ).require("elastic_modulus")
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

# Legacy analytical aero / unused input container (inactive).
#     def get_CNa(self, M: float, alpha: float):
#         A_plan = self.cfg["vehicle"]["OMLD"] * self.length
#         self.CNa = dist.weighted(aero.body_CNa(M, alpha, A_plan, self.ref_area), self.lat_area)

    def get_thermal_oml_area(self) -> np.ndarray:
        return self.surf_area.copy()

    def get_thermal_shell_mass(self) -> np.ndarray:
        return self.shell_mass.copy()

    def get_thermal_internal_area(self) -> np.ndarray:
        return np.full(self.n, self.copv.internal_area / self.n)
