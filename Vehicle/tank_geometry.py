"""Shared, immutable tank geometry for structure, fluid fill and thermal cells."""
from dataclasses import dataclass
from functools import cached_property
import numpy as np

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

@dataclass(frozen=True)
class PressTankGeometry:
    """Immutable COPV geometry used by the gas-volume node."""

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
        radius = 0.5 * self.inner_diameter
        head_depth = radius / self.ellipse_ratio
        resolution = max(512, self.resolution)
        dx = self.length / resolution
        x = (np.arange(resolution) + 0.5) * dx
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
        cross_area = np.pi * wall_radius**2
        wall_area = 2.0 * np.pi * wall_radius * np.sqrt(1.0 + slope**2) * dx
        return {
            "height": np.linspace(0.0, self.length, resolution + 1),
            "volume": np.concatenate(([0.0], np.cumsum(cross_area * dx))),
            "area": np.concatenate(([0.0], np.cumsum(wall_area))),
        }

    def fill_state(self, liquid_volume: float):
        """Return bottom-up liquid height and phase contact areas."""

        if not 0.0 <= liquid_volume <= self.volume:
            raise ValueError("Liquid volume must remain within the pressure tank")
        profile = self._profile
        target = liquid_volume / self.volume * profile["volume"][-1]
        fill_height = float(
            np.interp(target, profile["volume"], profile["height"])
        )
        profile_area = float(np.interp(target, profile["volume"], profile["area"]))
        liquid_area = self.internal_area * profile_area / profile["area"][-1]
        return {
            "fill_height": fill_height,
            "liquid_contact_area": liquid_area,
            "ullage_contact_area": self.internal_area - liquid_area,
        }
