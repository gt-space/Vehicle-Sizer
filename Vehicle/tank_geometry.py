"""Shared, immutable tank geometry for structure, fluid fill and thermal cells."""
from dataclasses import dataclass
from functools import cached_property
import numpy as np

def ellipsoidal_profile(length, inner_diameter, cylinder_length, ellipse_ratio, resolution,
                        passthrough_diameter=0., *, volume, internal_area=None):
    """Integrate the fixed internal wall and usable cross-section on one grid."""
    radius, tube = inner_diameter / 2, passthrough_diameter / 2
    head = radius / ellipse_ratio
    dx = length / resolution
    x = (np.arange(resolution) + .5) * dx
    wall_radius, slope = np.full_like(x, radius), np.zeros_like(x)
    for mask, center in ((x < head, head), (x > head + cylinder_length, head + cylinder_length)):
        axial = (x[mask] - center) / head
        root = np.sqrt(np.maximum(1 - axial**2, 0.))
        wall_radius[mask] = radius * root
        slope[mask] = -radius * axial / (head * root)
    cross_area = np.where(wall_radius > tube, np.pi * (wall_radius**2 - tube**2), 0.)
    wall_area = np.where(wall_radius > tube,
                         (2 * np.pi * wall_radius * np.sqrt(1 + slope**2) + 2 * np.pi * tube) * dx, 0.)
    volumes = np.concatenate(([0.], np.cumsum(cross_area * dx)))
    areas = np.concatenate(([0.], np.cumsum(wall_area)))
    if volumes[-1] <= 0.:
        raise ValueError("Tank geometry has no usable internal volume")
    volumes *= volume / volumes[-1]
    volumes[-1] = volume
    if internal_area is not None:
        areas *= internal_area / areas[-1]
        areas[-1] = internal_area
    return dict(length=length, height=np.linspace(0., length, resolution + 1),
                volume=volumes, area=areas)


class TankGeometry:
    """Shared fill interpolation on a cached physical-volume/contact-area profile."""

    def fill_state(self, liquid_volume: float):
        if not 0.0 <= liquid_volume <= self.volume:
            raise ValueError("Liquid volume must remain within the tank volume")
        profile = self._profile
        liquid_area = float(np.interp(liquid_volume, profile["volume"], profile["area"]))
        return dict(fill_height=float(np.interp(liquid_volume, profile["volume"], profile["height"])),
                    liquid_contact_area=liquid_area,
                    ullage_contact_area=float(profile["area"][-1]) - liquid_area)


@dataclass(frozen=True)
class PropTankGeometry(TankGeometry):
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
        length = self.cylinder_length + self.inner_diameter / self.ellipse_ratio
        return ellipsoidal_profile(length, self.inner_diameter, self.cylinder_length,
                                   self.ellipse_ratio, self.resolution, self.passthrough_diameter,
                                   volume=self.volume)

    @property
    def internal_area(self) -> float:
        """Total wetted wall area available for gas-volume heat transfer."""

        return float(self._profile["area"][-1])

    @property
    def axial_internal_area(self) -> np.ndarray:
        """Internal wall area in each axial geometry cell."""

        return np.diff(self._profile["area"])

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

@dataclass(frozen=True)
class PressTankGeometry(TankGeometry):
    """Immutable pressurant tank geometry used by the fluid network."""

    volume: float
    length: float
    inner_diameter: float
    cylinder_length: float
    ellipse_ratio: float
    internal_area: float
    resolution: int = 1

    def __post_init__(self):
        if any(
            value <= 0.0
            for value in (
                self.volume,
                self.length,
                self.inner_diameter,
                self.cylinder_length,
                self.ellipse_ratio,
                self.internal_area,
            )
        ):
            raise ValueError("Pressure tank geometry values must be positive")
        if self.resolution < 1:
            raise ValueError("Pressure tank geometry resolution must be positive")

    def axial_mass(self, mass: float) -> np.ndarray:
        """Return the local fore-to-aft gas mass vector."""

        if mass < 0.0:
            raise ValueError("Pressure-tank gas mass cannot be negative")
        return np.full(self.resolution, mass / self.resolution)

    @cached_property
    def _profile(self):
        return ellipsoidal_profile(self.length, self.inner_diameter, self.cylinder_length,
                                   self.ellipse_ratio, max(512, self.resolution),
                                   volume=self.volume, internal_area=self.internal_area)
