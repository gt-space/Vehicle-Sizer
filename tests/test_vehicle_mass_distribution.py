import unittest
from types import SimpleNamespace

import numpy as np

from Vehicle.Vehicle import Vehicle


class FakeTankSection:
    def __init__(self, tank_id, station, dry_mass):
        self.tank_id = tank_id
        self.station = np.asarray(station, dtype=float)
        self.dry_mass = np.asarray(dry_mass, dtype=float)
        self.mass = self.dry_mass.copy()
        self.EI = np.ones_like(self.mass)
        self.lat_area = np.ones_like(self.mass)
        self.surf_area = np.ones_like(self.mass)
        self.Ixx = 0.0
        self.Iyy = 0.0
        self.cg = float(np.mean(self.station))

    def set_fluid_mass(self, axial_mass):
        self.mass = self.dry_mass + np.asarray(axial_mass, dtype=float)
        self.cg = float(np.sum(self.mass * self.station) / np.sum(self.mass))


class VehicleMassDistributionTests(unittest.TestCase):
    def test_network_vectors_update_vehicle_mass_and_cg(self):
        forward_tank = FakeTankSection("ox_tank", [0.0, 1.0], [1.0, 1.0])
        aft_tank = FakeTankSection("fuel_tank", [2.0, 3.0], [1.0, 1.0])
        vehicle = Vehicle(
            cfg={"vehicle": {"dx": 1.0}},
            engine=SimpleNamespace(),
            sections=[forward_tank, aft_tank],
        )
        vehicle._assemble_vectors()
        vehicle.get_mass_properties()
        dry_cg = vehicle.cg

        vehicle.update_mass_distribution(
            {
                "oxidizer_node": {
                    "tank_id": "ox_tank",
                    "axial_mass": np.array([0.0, 2.0]),
                },
                "fuel_node": {
                    "tank_id": "fuel_tank",
                    "axial_mass": np.array([0.0, 4.0]),
                },
                "junction": {"P": 2.0e6},
            }
        )

        self.assertAlmostEqual(vehicle.total_mass, 10.0)
        self.assertGreater(vehicle.cg, dry_cg)
        np.testing.assert_allclose(vehicle.mass, [1.0, 3.0, 1.0, 5.0])


if __name__ == "__main__":
    unittest.main()
