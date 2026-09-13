import unittest
import sys
from unittest.mock import MagicMock

import numpy as np

sys.modules.setdefault("matproplib", MagicMock())
sys.modules.setdefault("Vehicle.utils.heating", MagicMock())

from Vehicle.Engine import Engine


class EngineTests(unittest.TestCase):
    def test_axial_mass_conserves_engine_mass(self):
        engine = Engine(mass=30.0, length=0.5, exit_area=0.02)

        distribution = engine.axial_mass(6)

        self.assertEqual(distribution.shape, (6,))
        self.assertAlmostEqual(float(np.sum(distribution)), engine.mass)

    def test_rejects_invalid_structural_properties(self):
        with self.assertRaises(ValueError):
            Engine(mass=-1.0, length=0.5, exit_area=0.02)
        with self.assertRaises(ValueError):
            Engine(mass=1.0, length=0.0, exit_area=0.02)
        with self.assertRaises(ValueError):
            Engine(mass=1.0, length=0.5, exit_area=0.0)


if __name__ == "__main__":
    unittest.main()
