import unittest
import numpy as np
from Configs.loader import load_config
from Vehicle.sections.InterTank import InterTank

class IntertankMassTests(unittest.TestCase):
    def test_manual_labels_and_override(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        section = InterTank(cfg, .3, 3.6e-5, stringer_unit_mass=2.7)
        section.start_station = 0
        section.station = section.local_centers
        section.mass_inputs = {'masses': {'avi': 2, 'plumbing': 5}}
        section.build()
        self.assertAlmostEqual(section.mass.sum(), 2.7 * .3 * 4 + 7)
        section.mass_inputs = {'mass_override': 20}
        section.build()
        self.assertAlmostEqual(section.mass.sum(), 20)
        self.assertTrue(np.all(section.EI > 0))

    def test_invalid_linear_mass(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        for value in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                InterTank(cfg, .3, 3.6e-5, stringer_unit_mass=value)
