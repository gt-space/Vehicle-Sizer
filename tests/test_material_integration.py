import unittest
import numpy as np
from Configs.loader import load_config
from Vehicle.sections.AviBay import AviBay
from Vehicle.sections.Nosecone import Nosecone
from Vehicle.sections.FinCan import FinCan
from Vehicle.Engine import Engine

class MaterialIntegrationTests(unittest.TestCase):
    def test_removed_structure_has_only_manual_mass(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        bay = AviBay(cfg)
        bay.station = bay.local_centers
        bay.mass_inputs = {'masses': {'box': 3, 'bulkhead': 2}}
        bay.build()
        self.assertAlmostEqual(bay.mass.sum(), 5)
        self.assertEqual(bay.shell_mass.sum(), 0)

    def test_nose_shell_and_fins_still_calculated(self):
        cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
        nose = Nosecone(cfg)
        nose.get_mass()
        original = nose.mass.sum()
        cfg['nosecone']['wall_thickness'] *= 2
        nose = Nosecone(cfg)
        nose.get_mass()
        self.assertAlmostEqual(nose.mass.sum(), original * 2)
        fin = FinCan(cfg, Engine(30, .5715, .01))
        fin.get_mass()
        self.assertGreater(fin.fin_shell_mass, 0)
        self.assertGreater(fin.boattail_shell_mass.sum(), 0)
