"""Pressurant section with empirical COPV or pressure-sized metal construction."""
import warnings

import numpy as np

from .Section import Section
from ..Material import MaterialProperties
from ..tank_geometry import PressTankGeometry
from ..utils import distribute as dist
from ..utils import geometry as geo


class PressTank(Section):

    def __init__(self, cfg: dict, tank_id: str):
        super().__init__(cfg)
        self.tank_id = tank_id
        definition = cfg["tanks"][tank_id]
        self.construction = definition.get("construction", "metal")
        self.volume = self.configured_volume(definition)
        self.diameter = float(definition["outer_diameter"])
        self.ellipse_ratio = float(definition["ellipse_ratio"])
        self.length_input = definition.get("length")
        self.mass_input = definition.get("mass")
        for name in ("length", "mass"):
            value = getattr(self, f"{name}_input")
            if value is not None:
                value = float(value)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"Pressurant tank {name} must be finite and positive")
                setattr(self, f"{name}_input", value)
        if (not np.isfinite([self.volume, self.diameter, self.ellipse_ratio]).all()
                or self.volume <= 0 or self.diameter <= 0 or self.ellipse_ratio <= 1):
            raise ValueError("Pressurant tank volume and diameter must be positive; ellipse_ratio > 1")
        self.wall_thickness, self.required_wall_thickness = self.wall_sizing(cfg, definition)
        if (self.required_wall_thickness is not None
                and self.wall_thickness < self.required_wall_thickness):
            warnings.warn(
                f"Tank '{tank_id}' wall thickness {self.wall_thickness:.6g} m is below "
                f"the required {self.required_wall_thickness:.6g} m",
                RuntimeWarning, stacklevel=2)
        if self.inner_diameter <= 0:
            raise ValueError("Pressurant tank wall thickness leaves no internal diameter")
        if self.construction == "copv":
            self.material_density = float(definition["equivalent_density"])
            self.wall_material = definition.get("material", cfg["nosecone"]["material"])
        else:
            self.material = MaterialProperties.from_name(definition["material"])
            self.material_density = self.material.density
            self.wall_material = self.material.name
        if not np.isfinite(self.material_density) or self.material_density <= 0:
            raise ValueError("Pressurant tank density must be finite and positive")
        self.cylinder_length = ((self.volume - (4 / 3) * np.pi * (self.inner_diameter / 2)**2 * self.head_depth)
                                / (np.pi * (self.inner_diameter / 2)**2)
                                if self.length_input is None else self.length_input - 2 * self.head_depth)
        if not np.isfinite(self.cylinder_length) or self.cylinder_length <= 0:
            raise ValueError("Pressurant tank volume/length must leave a positive cylinder between the endcaps")
        self.length = self.cylinder_length + 2 * self.head_depth
        self.set_grid()
        self.emissivity = 0.85

    @staticmethod
    def configured_volume(definition):
        if "volume" in definition and "volume_liters" in definition:
            raise ValueError("Specify pressurant volume (m^3) or volume_liters, not both")
        return (float(definition["volume"]) if "volume" in definition
                else float(definition["volume_liters"]) * 1e-3)

    @staticmethod
    def wall_sizing(cfg, definition):
        """Return used and required gauges; shared by assembly and optimizer."""
        construction = definition.get("construction", "metal")
        diameter = float(definition["outer_diameter"])
        required = None
        if construction == "copv":
            slope = float(definition["thickness_slope"])
            intercept = float(definition["thickness_intercept"])
            if not np.isfinite([diameter, slope, intercept]).all() or diameter <= 0 or slope < 0 or intercept < 0:
                raise ValueError("COPV diameter must be positive and thickness coefficients nonnegative")
            wall = slope * diameter + intercept
        elif construction == "metal":
            advanced = cfg.get("advanced", {})
            required = geo.pressure_wall_thickness(
                float(definition["design_pressure"]), diameter,
                float(definition.get("pressure_fos", advanced.get("tank_pressure_fos"))),
                float(definition.get("weld_allowable", advanced.get("weld_allowable"))))
            wall = required
        else:
            raise ValueError(f"Unknown pressurant construction {construction!r}; use copv or metal")
        supplied = definition.get("wall_thickness")
        if supplied is not None:
            wall = float(supplied)
        if not np.isfinite(wall) or wall <= 0:
            raise ValueError("Pressurant tank wall thickness must be finite and positive")
        return wall, required

    @property
    def inner_diameter(self):
        return self.diameter - 2 * self.wall_thickness

    @property
    def head_depth(self):
        return self.inner_diameter / (2 * self.ellipse_ratio)

    @property
    def shell_volume(self):
        if self.construction == "copv":
            # Preserve the empirical COPV mass model using nested heads.
            outer_radius, inner_radius = self.diameter / 2, self.inner_diameter / 2
            return (geo.annulus_volume(outer_radius, inner_radius, self.cylinder_length)
                    + 4 * np.pi * (outer_radius**3 - inner_radius**3) / (3 * self.ellipse_ratio))
        return geo.tank_shell_volume(
            self.diameter, self.wall_thickness, self.cylinder_length, self.ellipse_ratio,
            float(self.cfg["advanced"]["endcap_mass_multiplier"]))

    @property
    def internal_area(self):
        radius = self.inner_diameter / 2
        eccentricity = np.sqrt(1 - (self.head_depth / radius)**2)
        end_area = 2 * np.pi * radius**2 * (
            1 + (1 - eccentricity**2) / eccentricity * np.arctanh(eccentricity))
        return 2 * np.pi * radius * self.cylinder_length + end_area

    def _get_dry_mass(self):
        return (self.mass_input if self.mass_input is not None
                else self.shell_volume * self.material_density)

    def get_mass(self):
        self.dry_mass = dist.uniform(self._get_dry_mass(), self.n)
        self.shell_mass = self.dry_mass.copy()
        self.mass = self.dry_mass.copy()

    def get_fluid_geometry(self) -> PressTankGeometry:
        """Export immutable internal geometry for the fluid network."""
        return PressTankGeometry(
            volume=self.volume, length=self.length, inner_diameter=self.inner_diameter,
            cylinder_length=self.cylinder_length, ellipse_ratio=self.ellipse_ratio,
            internal_area=self.internal_area, resolution=self.n)

    def set_fluid_mass(self, axial_mass: np.ndarray) -> None:
        """Add a network-supplied gas vector to this tank's dry mass."""
        axial_mass = np.asarray(axial_mass, dtype=float)
        if axial_mass.shape != self.dry_mass.shape:
            raise ValueError(f"Tank '{self.tank_id}' requires {self.n} axial mass values")
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

    def get_thermal_oml_area(self):
        return self.surf_area.copy()

    def get_thermal_shell_mass(self):
        return self.shell_mass.copy()

    def get_thermal_internal_area(self):
        return np.full(self.n, self.internal_area / self.n)
