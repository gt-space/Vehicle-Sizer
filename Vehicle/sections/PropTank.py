import numpy as np
import warnings
from dataclasses import dataclass
from typing import Optional
from functools import cached_property
from .Section import Section
from ..Material import MaterialProperties
from ..utils import distribute as dist
from ..utils import aero
from ..utils import geometry as geo
from ..utils import heating

@dataclass(frozen=True)
class PropTankGeometry:
    """Immutable internal geometry used to derive fill-dependent properties."""

    volume: float
    inner_diameter: float
    cylinder_length: float
    ellipse_ratio: float
    passthrough_diameter: float
    resolution: int = 512

    def __post_init__(self):
        positive = (
            self.volume,
            self.inner_diameter,
            self.cylinder_length,
            self.ellipse_ratio,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("Propellant tank geometry values must be positive")
        if not 0.0 <= self.passthrough_diameter < self.inner_diameter:
            raise ValueError("Passthrough diameter must be smaller than the tank")
        if self.resolution < 2:
            raise ValueError("Tank geometry resolution must be at least two")

    @cached_property
    def _profile(self):
        radius = 0.5 * self.inner_diameter
        passthrough_radius = 0.5 * self.passthrough_diameter
        head_depth = radius / self.ellipse_ratio
        length = self.cylinder_length + 2.0 * head_depth
        dx = length / self.resolution
        x = (np.arange(self.resolution) + 0.5) * dx

        wall_radius = np.full_like(x, radius)
        slope = np.zeros_like(x)
        lower = x < head_depth
        upper = x > head_depth + self.cylinder_length
        for mask, center in (
            (lower, head_depth),
            (upper, head_depth + self.cylinder_length),
        ):
            axial = (x[mask] - center) / head_depth
            root = np.sqrt(np.maximum(1.0 - axial**2, 0.0))
            wall_radius[mask] = radius * root
            slope[mask] = -radius * axial / (head_depth * root)

        open_section = wall_radius > passthrough_radius
        cross_area = np.where(
            open_section,
            np.pi * (wall_radius**2 - passthrough_radius**2),
            0.0,
        )
        wall_area = np.where(
            open_section,
            (
                2.0 * np.pi * wall_radius * np.sqrt(1.0 + slope**2)
                + 2.0 * np.pi * passthrough_radius
            )
            * dx,
            0.0,
        )
        cell_volume = cross_area * dx
        return {
            "length": length,
            "volume": np.concatenate(([0.0], np.cumsum(cell_volume))),
            "area": np.concatenate(([0.0], np.cumsum(wall_area))),
            "height": np.linspace(0.0, length, self.resolution + 1),
        }

    def fill_state(self, liquid_volume: float):
        """Return fill height and liquid/ullage wall contact areas."""

        if not 0.0 <= liquid_volume <= self.volume:
            raise ValueError("Liquid volume must remain within the tank volume")
        profile = self._profile
        geometric_volume = profile["volume"][-1]
        if geometric_volume <= 0.0:
            raise ValueError("Tank geometry has no usable internal volume")
        target = liquid_volume / self.volume * geometric_volume
        height = float(np.interp(target, profile["volume"], profile["height"]))
        liquid_area = float(np.interp(target, profile["volume"], profile["area"]))
        total_area = float(profile["area"][-1])
        return {
            "fill_height": height,
            "liquid_contact_area": liquid_area,
            "ullage_contact_area": total_area - liquid_area,
        }

    @property
    def internal_area(self) -> float:
        """Total wetted wall area available for gas-volume heat transfer."""

        return float(self._profile["area"][-1])

    def axial_mass(
        self,
        liquid_volume: float,
        liquid_mass: float,
        ullage_mass: float,
    ) -> np.ndarray:
        """Return the local fore-to-aft fluid mass vector."""

        if not 0.0 <= liquid_volume <= self.volume:
            raise ValueError("Liquid volume must remain within the tank volume")
        if liquid_mass < 0.0 or ullage_mass < 0.0:
            raise ValueError("Tank phase masses cannot be negative")

        cell_volume = np.diff(self._profile["volume"])
        cell_volume *= self.volume / np.sum(cell_volume)
        aft_volume = cell_volume[::-1]
        volume_before = np.concatenate(([0.0], np.cumsum(aft_volume)[:-1]))
        liquid_cell_volume = np.clip(
            liquid_volume - volume_before,
            0.0,
            aft_volume,
        )[::-1]
        ullage_cell_volume = cell_volume - liquid_cell_volume

        def distribute(mass: float, volume: np.ndarray, phase: str) -> np.ndarray:
            total_volume = np.sum(volume)
            if mass == 0.0:
                return np.zeros(self.resolution)
            if total_volume <= 0.0:
                raise ValueError(f"{phase} mass requires nonzero {phase} volume")
            return mass * volume / total_volume

        return distribute(
            liquid_mass, liquid_cell_volume, "liquid"
        ) + distribute(ullage_mass, ullage_cell_volume, "ullage")

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
        self.n = int(np.ceil(self.length / self.dx))

    def get_mass(self):
        dry_mass = self._get_dry_mass()
        self.dry_mass = dist.uniform(dry_mass, self.n)
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

        allowable = self.material.require("yield_strength")
        ratio = 1.5 * self.max_pressure / allowable
        required = max(
            ratio * (0.5 * self.OMLD) / (1.0 + ratio),
            self.t_wall_min,
        )
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

    def get_CNa(self, M: float, alpha: float):
        A_plan = self.cfg["vehicle"]["OMLD"] * self.length
        self.CNa = dist.weighted(aero.body_CNa(M, alpha, A_plan, self.ref_area), self.lat_area)

    def get_heat_flux(self, atm, theta: float):
        self.heat_flux = heating.get_body_heating(self.station, self.wall_temp, atm, theta)
