import unittest

import numpy as np

from Flight.loads import Loads


class FakeVehicle:
    station = np.array([0.5, 1.5, 2.5])
    cell_edges = np.array([0.0, 1.0, 2.0, 3.0])
    engine_start_station = 1.0
    length = 3.0
    mass = np.array([1.0, 2.0, 1.0])
    total_mass = 4.0
    cg = 1.5
    Iyy = 2.0
    EI = np.full(3, 100.0)


class FakeAero:
    reference_area = 1.0

    @staticmethod
    def axial_distribution(mach, alpha, engine_on):
        base = -.1 if engine_on else .1
        # Deliberately coarser than the vehicle grid: base spike spans 1.5 m.
        x = np.array([0., 1.5, 3.])
        spike = np.array([0., 0., 2*base/1.5])
        return dict(x=x, dca_dx=np.full(3, .2/3) + spike, parts={"base": spike},
                    point_loads={"base": (3., base)}, ca=.2+base)

    @staticmethod
    def normal_distribution(mach, alpha):
        del mach, alpha
        x = np.linspace(0.0, 3.0, 301)
        return {"x": x, "dcn_dx": 2.0 * x / 9.0}


class LoadsTests(unittest.TestCase):
    def setUp(self):
        self.loads = Loads(FakeVehicle(), FakeAero())

    def test_axial_resultant_uses_ca_force_without_drag_split(self):
        internal = self.loads.get_axial_load(axial_forces=np.array([3., 0., 0.]), thrust=9.0)
        np.testing.assert_allclose(internal, [4.5, -1.5, 0.0])

    def test_thrust_follows_interface_and_preserves_force_balance(self):
        self.loads.vehicle.engine_start_station = 2.0
        internal = self.loads.get_axial_load(np.array([3., 0., 0.]), thrust=9.0)
        np.testing.assert_allclose(internal, [4.5, 7.5, 0.0])
        self.loads.vehicle.engine_start_station = 0.0
        internal = self.loads.get_axial_load(np.array([3., 0., 0.]), thrust=9.0)
        np.testing.assert_allclose(internal, [-4.5, -1.5, 0.0])

    def test_invalid_thrust_interface_is_not_clamped_to_tail(self):
        for station in (-1, 3, float("nan")):
            with self.subTest(station=station):
                self.loads.vehicle.engine_start_station = station
                with self.assertRaisesRegex(ValueError, "thrust interface"):
                    self.loads.get_axial_load(np.array([3., 0., 0.]), 9)

    def test_base_is_counted_once_and_only_in_last_cell(self):
        off = self.loads.get_axial_forces(10, .5, .1, False)
        on = self.loads.get_axial_forces(10, .5, .1, True)
        np.testing.assert_allclose(off, [2/3, 2/3, 2/3+1])
        np.testing.assert_allclose(on, [2/3, 2/3, 2/3-1])
        np.testing.assert_allclose(off-on, [0, 0, 2])
        self.assertAlmostEqual(off.sum(), 3)
        self.assertAlmostEqual(on.sum(), 1)

    def test_piecewise_linear_remap_is_exact_on_misaligned_cells(self):
        self.loads.vehicle.cell_edges = np.array([0., .2, 2.7, 3.])
        mapped = self.loads._cell_coefficients([0., 1.5, 3.], [0., 3., 6.])
        np.testing.assert_allclose(mapped, np.diff(self.loads.vehicle.cell_edges**2))

    def test_normal_inertia_loads_balance_force_and_moment(self):
        force = self.loads.get_normal_load(q=10.0, M=0.5, alpha=0.1)
        self.assertAlmostEqual(np.sum(force), 0.0)
        self.assertAlmostEqual(np.sum(force * (FakeVehicle.station - FakeVehicle.cg)), 0.0)

        result = self.loads.evaluate(10.0, 0.5, 0.1, axial_force=3.0, thrust=9.0, engine_on=False)
        self.assertAlmostEqual(result["axial"][-1], 0.0)
        self.assertAlmostEqual(result["shear"][-1], 0.0)
        self.assertAlmostEqual(result["bending"][-1], 0.0)


if __name__ == "__main__":
    unittest.main()
