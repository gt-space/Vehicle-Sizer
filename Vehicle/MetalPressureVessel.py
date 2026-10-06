"""Metal pressurant vessel using material density and pressure-sized walls."""
from dataclasses import dataclass
import numpy as np
from .COPV import COPV

@dataclass(frozen=True)
class MetalPressureVessel(COPV):
    design_pressure: float = 0.0
    pressure_fos: float = 1.5
    allowable_stress: float = 0.0

    def __post_init__(self):
        if (not np.isfinite([self.design_pressure, self.pressure_fos, self.allowable_stress]).all()
                or self.design_pressure <= 0 or self.allowable_stress <= 0 or self.pressure_fos < 1):
            raise ValueError("Metal vessel requires positive design_pressure/allowable_stress and pressure_fos >= 1")
        super().__post_init__()

    @property
    def wall_thickness(self):
        return self.pressure_fos * self.design_pressure * self.diameter / (2 * self.allowable_stress)
