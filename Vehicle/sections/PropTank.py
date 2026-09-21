import numpy as np
import warnings
from typing import Optional
from .Section import Section
from ..Material import MaterialProperties
from ..utils import distribute as dist
# from ..utils import aero  # Legacy analytical aero disabled.
from ..utils import geometry as geo

TANK_STOCK_THICKNESSES_IN = np.array([
    0.040,
    0.050,
    0.063,   # ~1/16"
    0.080,
    0.090,
    0.100,
    0.125,   # 1/8"
    0.160,
    0.190,
    0.250,   # 1/4"
])
TANK_STOCK_THICKNESSES = TANK_STOCK_THICKNESSES_IN * 0.0254

from ..tank_geometry import PropTankGeometry

class PropTank(Section):

    def __init__(
        self,
        cfg: dict,
        prop_mass: float,
        liquid_density: float,
        material: MaterialProperties,
        wall_thickness: Optional[float],
        max_pressure: float,
        t_wall_min: float,
        passthrough_diameter: float,
        passthrough_wall_thickness: float,
        ellipse_ratio: float,
        ullage_factor: float,
        tank_id: str,
        weld_efficiency: float = 1.0,
    ):

        super().__init__(cfg)
        self.tank_id = tank_id
        self.passthrough_diameter = float(passthrough_diameter)
        self.ellipse_ratio = float(ellipse_ratio)
        self.ullage_factor = float(ullage_factor)
        self.OMLD = float(cfg["vehicle"]["OMLD"])
        self.prop_mass = float(prop_mass)
        self.material = material
        self.wall_material = material.name
        self.max_pressure = float(max_pressure)
        self.t_wall_min = float(t_wall_min)
        self.weld_efficiency = float(weld_efficiency)
        if not 0.0 < self.weld_efficiency <= 1.0:
            raise ValueError("Tank weld_efficiency must be in (0, 1]")
        self.wall_thickness = self.get_thickness(wall_thickness)
        self.passthrough_wall_thickness = float(passthrough_wall_thickness)
        self.emissivity = 0.85
        self.liquid_density = float(liquid_density)

        if self.prop_mass <= 0.0:
            raise ValueError("Propellant mass must be positive")
        if self.max_pressure <= 0.0 or self.t_wall_min <= 0.0:
            raise ValueError("Tank pressure and minimum wall gauge must be positive")
        if self.passthrough_diameter > 0.0 and self.passthrough_wall_thickness <= 0.0:
            raise ValueError("Passthrough wall thickness must be positive")
        if self.liquid_density <= 0.0:
            raise ValueError("Initial propellant density must be positive")
        if self.ullage_factor <= 1.0:
            raise ValueError("Ullage factor must be greater than one")
        if self.ellipse_ratio <= 1.0:
            raise ValueError("Tank ellipse ratio must be greater than one")

        self.TankVolume = self._tank_volume()
        self._get_length()
        self.set_grid()

    def get_mass(self):
        dry_mass = self._get_dry_mass()
        self.dry_mass = dist.uniform(dry_mass, self.n)
        self.shell_mass = self.dry_mass.copy()
        self.mass = self.dry_mass

    def _get_dry_mass(self) -> float:
        D = self.OMLD
        D_pass = self.passthrough_diameter
        t = self.wall_thickness
        t_pass = self.passthrough_wall_thickness
        e = self.ellipse_ratio
        rho = self.material.density

        k = (
            2 * e
            + (1 / np.sqrt(e**2 - 1))
            * np.log((e + np.sqrt(e**2 - 1)) / (e - np.sqrt(e**2 - 1)))
        )

        V_end = ((1/4) * np.pi * (D - 2*t) * t * k) / (2 * e)
        V_cyl = geo.annulus_volume(D * 0.5, D * 0.5 - t, self.cyl_length)
        if D_pass > 0.0 and t_pass >= 0.5 * D_pass:
            raise ValueError("Passthrough wall consumes its internal diameter")
        V_pass = (
            geo.annulus_volume(
                D_pass * 0.5,
                D_pass * 0.5 - t_pass,
                self.cyl_length,
            )
            if D_pass > 0.0
            else 0.0
        )

        return rho * (V_end + V_cyl + V_pass)

    def get_thickness(self, supplied: Optional[float] = None) -> float:
        """Return supplied gauge or pressure-size it from the fixed OML diameter."""

        allowable = self.material.require("yield_strength") * self.weld_efficiency
        ratio = 1.5 * self.max_pressure / allowable
        required = max(
            ratio * (0.5 * self.OMLD) / (1.0 + ratio),
            self.t_wall_min,
        )
        self.required_wall_thickness = required
        if supplied is None:
            return required
        supplied = float(supplied)
        if supplied <= 0.0:
            raise ValueError("Tank wall thickness must be positive")
        if supplied < required:
            warnings.warn(
                f"Tank '{self.tank_id}' wall thickness {supplied:.6g} m is below "
                f"the required {required:.6g} m",
                RuntimeWarning,
                stacklevel=2,
            )
        return supplied

    def _tank_volume(self) -> float:
        return self._liquid_capacity() * self.ullage_factor

    def _liquid_capacity(self) -> float:
        return self.prop_mass / self.liquid_density

    def get_fluid_geometry(self) -> PropTankGeometry:
        """Export immutable geometry for the fluid-network tank node."""

        return PropTankGeometry(
            volume=self.volume,
            inner_diameter=self.OMLD - 2.0 * self.wall_thickness,
            cylinder_length=self.cyl_length,
            ellipse_ratio=self.ellipse_ratio,
            passthrough_diameter=self.passthrough_diameter,
            resolution=self.n,
        )

    def set_fluid_mass(self, axial_mass: np.ndarray) -> None:
        """Add a network-supplied fluid vector to this tank's dry mass."""

        axial_mass = np.asarray(axial_mass, dtype=float)
        if axial_mass.shape != self.dry_mass.shape:
            raise ValueError(
                f"Tank '{self.tank_id}' requires {self.n} axial mass values"
            )
        if np.any(axial_mass < 0.0):
            raise ValueError(f"Tank '{self.tank_id}' fluid mass cannot be negative")
        self.mass = self.dry_mass + axial_mass
        self.get_MOI()

    def _get_length(self):
        inner_diameter = self.OMLD - 2.0 * self.wall_thickness
        if inner_diameter <= 0.0:
            raise ValueError("Tank wall thickness leaves no internal diameter")
        if not 0.0 <= self.passthrough_diameter < inner_diameter:
            raise ValueError("Passthrough diameter must be smaller than tank ID")

        self.volume = self._tank_volume()
        radius = 0.5 * inner_diameter
        pass_radius = 0.5 * self.passthrough_diameter
        head_depth = radius / self.ellipse_ratio
        beta = np.sqrt(1.0 - (pass_radius / radius) ** 2)

        # Usable volume in both ellipsoidal heads, excluding the axial
        # passthrough tube. The remaining volume is carried by the cylinder.
        head_volume = (
            (4.0 / 3.0) * np.pi * radius**2 * head_depth * beta**3
        )
        cylinder_area = np.pi * (radius**2 - pass_radius**2)
        self.cyl_length = (self.volume - head_volume) / cylinder_area
        if self.cyl_length <= 0.0:
            raise ValueError(
                "Requested tank volume is smaller than its endcap volume"
            )
        self.length = self.cyl_length + 2.0 * head_depth

    def get_EI(self):
        r_o = self.OMLD * 0.5
        r_i = r_o - self.wall_thickness
        E = self.material.require("elastic_modulus")
        self.EI = dist.uniform_full(E * geo.annulus_second_moment(r_o, r_i), self.n)

    def get_area(self):
        r = self.OMLD * 0.5
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
        area = self.get_thermal_internal_area()
        return np.sum(self.shell_mass) * area / np.sum(area)

    def get_thermal_internal_area(self) -> np.ndarray:
        return self.get_fluid_geometry().axial_internal_area
