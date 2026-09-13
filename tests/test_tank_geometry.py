import unittest
import sys
from unittest.mock import MagicMock

import numpy as np
from CoolProp.CoolProp import PropsSI

sys.modules.setdefault("matproplib", MagicMock())
sys.modules.setdefault("Vehicle.utils.heating", MagicMock())

from Vehicle.sections.PressTank import PressTankGeometry
from Vehicle.sections.PropTank import PropTank, PropTankGeometry
from Vehicle.COPV import COPV
from Vehicle.Material import MaterialProperties


class TankGeometryTests(unittest.TestCase):
    def test_propellant_wall_is_pressure_sized_when_not_supplied(self):
        material = MaterialProperties("test", 2700.0, 276.0e6, 77.0e9)
        pressure = 2.7e6
        tank = PropTank(
            cfg={"vehicle": {"dx": 0.01, "OMLD": 0.3048}},
            prop_mass=60.0,
            liquid_density=1000.0,
            material=material,
            wall_thickness=None,
            max_pressure=pressure,
            t_wall_min=0.001,
            passthrough_diameter=0.0,
            passthrough_wall_thickness=0.00254,
            ellipse_ratio=1.75,
            ullage_factor=1.10,
            tank_id="tank",
        )
        ratio = 1.5 * pressure / material.yield_strength
        expected = ratio * (0.5 * 0.3048) / (1.0 + ratio)
        self.assertAlmostEqual(tank.wall_thickness, expected)

    def test_copv_calculates_length_and_internal_area(self):
        copv = COPV(
            volume=0.02,
            mass=18.2,
            diameter=0.22,
            wall_thickness=0.0032,
            ellipse_ratio=1.75,
        )
        radius = 0.5 * copv.inner_diameter
        recovered_volume = (
            np.pi * radius**2 * copv.cylinder_length
            + 4.0 / 3.0 * np.pi * radius**2 * copv.head_depth
        )
        self.assertAlmostEqual(recovered_volume, copv.volume)
        self.assertGreater(copv.internal_area, 0.0)

    def test_propellant_mass_ullage_and_endcaps_set_tank_length(self):
        pressure = 2.6e6
        temperature = 95.0
        liquid_volume = 0.06
        density = PropsSI("Dmass", "P", pressure, "T", temperature, "Oxygen")
        tank = PropTank(
            cfg={"vehicle": {"dx": 0.01, "OMLD": 0.3048}},
            prop_mass=density * liquid_volume,
            liquid_density=density,
            material=MaterialProperties(
                "aluminum_6061_t6", 2700.0, 276.0e6, 77.0e9
            ),
            wall_thickness=0.0032,
            max_pressure=2.7e6,
            t_wall_min=0.001,
            passthrough_diameter=0.04,
            passthrough_wall_thickness=0.00254,
            ellipse_ratio=1.75,
            ullage_factor=1.10,
            tank_id="ox_tank",
        )

        radius = 0.5 * (0.3048 - 2.0 * 0.0032)
        pass_radius = 0.02
        head_depth = radius / 1.75
        beta = np.sqrt(1.0 - (pass_radius / radius) ** 2)
        head_volume = 4.0 / 3.0 * np.pi * radius**2 * head_depth * beta**3
        cylinder_volume = (
            np.pi * (radius**2 - pass_radius**2) * tank.cyl_length
        )

        self.assertAlmostEqual(tank.volume, liquid_volume * 1.10)
        self.assertAlmostEqual(head_volume + cylinder_volume, tank.volume)
        self.assertAlmostEqual(tank.length, tank.cyl_length + 2.0 * head_depth)

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
            inner_diameter=0.25,
            cylinder_length=0.5,
            ellipse_ratio=1.25,
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
            inner_diameter=0.25,
            cylinder_length=0.5,
            ellipse_ratio=1.25,
            internal_area=0.6,
            resolution=7,
        )

        axial_mass = geometry.axial_mass(3.5)

        self.assertEqual(len(axial_mass), 7)
        self.assertAlmostEqual(np.sum(axial_mass), 3.5)


if __name__ == "__main__":
    unittest.main()
