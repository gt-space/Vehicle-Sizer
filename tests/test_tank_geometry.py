import unittest
import sys
from unittest.mock import MagicMock

import numpy as np

sys.modules.setdefault("matproplib", MagicMock())
sys.modules.setdefault("Vehicle.utils.heating", MagicMock())

from Vehicle.sections.PressTank import PressTankGeometry
from Vehicle.sections.PropTank import PropTankGeometry


class TankGeometryTests(unittest.TestCase):
    def test_propellant_geometry_returns_fill_dependent_contact_areas(self):
        geometry = PropTankGeometry(
            volume=0.1,
            inner_diameter=0.4,
            cylinder_length=0.6,
            ellipse_ratio=1.5,
            passthrough_diameter=0.05,
        )

        empty = geometry.fill_state(0.0)
        half = geometry.fill_state(0.05)
        full = geometry.fill_state(0.1)

        self.assertEqual(empty["liquid_contact_area"], 0.0)
        self.assertEqual(full["ullage_contact_area"], 0.0)
        self.assertGreater(half["fill_height"], empty["fill_height"])
        self.assertLess(half["fill_height"], full["fill_height"])
        self.assertAlmostEqual(
            half["liquid_contact_area"] + half["ullage_contact_area"],
            full["liquid_contact_area"],
        )

    def test_press_tank_geometry_stores_internal_contact_area(self):
        geometry = PressTankGeometry(
            volume=0.02,
            length=0.7,
            diameter=0.25,
            internal_area=0.6,
        )

        self.assertEqual(geometry.internal_area, 0.6)

    def test_propellant_axial_mass_is_conserved_and_settles_aft(self):
        geometry = PropTankGeometry(
            volume=0.1,
            inner_diameter=0.4,
            cylinder_length=0.6,
            ellipse_ratio=1.5,
            passthrough_diameter=0.05,
            resolution=20,
        )

        axial_mass = geometry.axial_mass(
            liquid_volume=0.025,
            liquid_mass=20.0,
            ullage_mass=0.0,
        )

        self.assertEqual(len(axial_mass), 20)
        self.assertAlmostEqual(np.sum(axial_mass), 20.0)
        self.assertEqual(np.sum(axial_mass[:10]), 0.0)
        self.assertGreater(np.sum(axial_mass[10:]), 0.0)

    def test_pressure_tank_axial_mass_is_conserved(self):
        geometry = PressTankGeometry(
            volume=0.02,
            length=0.7,
            diameter=0.25,
            internal_area=0.6,
            resolution=7,
        )

        axial_mass = geometry.axial_mass(3.5)

        self.assertEqual(len(axial_mass), 7)
        self.assertAlmostEqual(np.sum(axial_mass), 3.5)


if __name__ == "__main__":
    unittest.main()
