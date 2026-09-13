from dataclasses import dataclass

import numpy as np

@dataclass
class COPV:
    volume: float
    mass: float
    length: float
    diameter: float
    max_pressure: float

    @property
    def internal_area(self):
        r = self.diameter * 0.5
        sphere_volume = (4 / 3) * np.pi * r**3
        cylinder_length = (self.volume - sphere_volume) / (np.pi * r**2)
        cylinder_area = 2 * np.pi * r * cylinder_length
        endcap_area = 4 * np.pi * r**2
        return cylinder_area + endcap_area

SPECTRONIK_20L_TYPE3 = COPV(0.02, 7, 0.661, 0.233, 350e5)
EATON_62803 = COPV(0.0213, 9.1, 0.738, 0.225, 227e5)
EATON_7173 = COPV(0.0226, 9.9, 0.662, 0.2642, 428e5)
EATON_62805 = COPV(0.023, 9.6, 0.7783, 0.225, 227e5)
EATON_6289 = COPV(0.0266, 13.6, 0.7831, 0.2436, 310e5)
EATON_6366 = COPV(0.0266, 18.8, 0.7887, 0.2586, 414e5)
EATON_7130 = COPV(0.0283, 12.7, 1.2195, 0.2642, 414e5)
EATON_6208 = COPV(0.0313, 18.2, 1.2195, 0.2134, 345e5)
EATON_6364 = COPV(0.0313, 22, 1.2256, 0.2184, 414e5)
EATON_7258 = COPV(0.045, 11.5, 1.0668, 0.2573, 345e5)
EATON_7226 = COPV(0.0452, 18.4, 1.0772, 0.2675, 379e5)