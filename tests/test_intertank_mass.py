import unittest
import numpy as np
from Configs.loader import load_config
from Vehicle.sections.InterTank import InterTank

class IntertankMassTests(unittest.TestCase):
    def test_manual_labels_and_override(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        section = InterTank(cfg, .3, 3.6e-5)
        section.start_station = 0
        section.station = section.local_centers
        section.mass_inputs = {'masses': {'stringers': 3, 'avi': 2, 'plumbing': 5}}
        section.build()
        self.assertAlmostEqual(section.mass.sum(), 10)
        section.mass_inputs = {'mass_override': 20}
        section.build()
        self.assertAlmostEqual(section.mass.sum(), 20)
        self.assertTrue(np.all(section.EI > 0))

    def test_no_implicit_hardware_mass(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        section = InterTank(cfg, .3, 3.6e-5)
        section.get_mass()
        np.testing.assert_array_equal(section.mass, np.zeros(section.n))
