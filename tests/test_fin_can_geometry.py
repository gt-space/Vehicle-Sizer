import unittest

import numpy as np

from Vehicle.Engine import Engine
from Vehicle.Vehicle import Vehicle
from Vehicle.sections.FinCan import FinCan


class FinCanGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "vehicle": {"OMLD": 0.3048, "dx": 0.01},
            "fin_can": {
                "span": 0.1905,
                "root_chord": 0.4572,
                "tip_chord": 0.1524,
                "sweep_fraction": 0.75,
                "fin_thickness": 0.009525,
                "fin_count": 4,
                "boattail_aft_diameter": 0.2667,
                "boattail_length": 0.5588,
                "boattail_wall_thickness": 0.00318,
                "material": "carbon_fiber_standard",
            },
        }
        exit_area = np.pi * (0.2286 / 2.0) ** 2
        self.engine = Engine(mass=30.0, length=0.5588, exit_area=exit_area)
        self.fin_can = FinCan(self.cfg, self.engine)
        self.fin_can.start_station = 0.0
        self.fin_can.end_station = self.fin_can.length
        self.fin_can.station = np.arange(self.fin_can.n) * self.fin_can.dx
        self.fin_can.get_area()

    def test_fin_mass_and_area_use_trapezoid_dimensions(self):
        expected_area = 0.5 * (0.4572 + 0.1524) * 0.1905
        self.assertAlmostEqual(self.fin_can.fin_area, expected_area)
        self.assertAlmostEqual(
            np.sum(self.fin_can.surf_area_fins),
            2.0 * self.fin_can.fin_count * expected_area,
        )

    def test_vehicle_exports_same_geometry_in_inches(self):
        vehicle = Vehicle.__new__(Vehicle)
        vehicle.cfg = {
            "vehicle": {"OMLD": 0.3048},
            "nosecone": {"fineness_ratio": 5.0},
        }
        vehicle.engine = self.engine
        vehicle.length = 9.144
        vehicle.sections = [self.fin_can]

        candidate = vehicle.aero_candidate()

        self.assertAlmostEqual(candidate["omld"], 12.0)
        self.assertAlmostEqual(candidate["length"], 360.0)
        self.assertAlmostEqual(candidate["exit"], 9.0)
        self.assertAlmostEqual(candidate["span"], 7.5)
        self.assertAlmostEqual(candidate["root"], 18.0)
        self.assertAlmostEqual(candidate["tip"], 6.0)


if __name__ == "__main__":
    unittest.main()
