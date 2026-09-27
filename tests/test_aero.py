import unittest

import numpy as np
from scipy.integrate import trapezoid

from Flight.flight_forces import Aero
from simulation_types import AtmosState, KinematicsState


CANDIDATE = {
    "omld": 12.0,
    "length": 360.0,
    "fineness": 5.0,
    "exit": 9.0,
    "boattail_aft": 10.5,
    "boattail_length": 22.0,
    "span": 7.5,
    "root": 18.0,
    "tip": 6.0,
    "sweep_fraction": 0.75,
    "thickness": 0.375,
}


class FakeDragModel:
    mach = np.array([0.1, 1.0])
    alpha = np.array([0.0, 10.0])
    ranges = {"exit": (5.9, "boattail_aft"), "length": (276.0, 500.0)}

    def check(self, candidate):
        for name, (low, high) in self.ranges.items():
            high = candidate[high] if isinstance(high, str) else high
            if not low <= candidate[name] <= high:
                raise ValueError(f"{name} outside model bounds")

    def __init__(self):
        self.table_calls = []

    def cd_table(self, candidate, nose="vonkarman", finish="10um", power_on=False):
        self.table_calls.append(("cd", power_on, dict(candidate), nose, finish))
        return np.array([[0.20, 0.30], [0.40, 0.50]]) + (0.0 if power_on else 0.10)

    def cn_table(self, candidate, nose="vonkarman"):
        self.table_calls.append(("cn", dict(candidate), nose))
        return np.array([[0.0, 0.20], [0.0, 0.40]])

    def cp_table(self, candidate, nose="vonkarman", fins_on_boattail=False):
        self.table_calls.append(("cp", dict(candidate), nose, fins_on_boattail))
        return np.array([[100.0, 110.0], [120.0, 130.0]])

    def cn_distribution(self, candidate, mach, alpha, **options):
        cn = 0.02 * alpha
        return {
            "x": np.array([0.0, candidate["length"]]),
            "dcn_dx": np.full(2, cn / candidate["length"]),
            "parts": {},
            "running_cn": np.array([0.0, cn]),
            "cn": cn,
            "cp": candidate["length"] / 2.0,
            "mach": mach,
            "alpha": alpha,
        }


class AeroTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeDragModel()
        self.aero = Aero(
            {
                "nose": "vonkarman",
                "finish": "20um",
                "fins_on_boattail": True,
                "aoa_schedule": [[0.0, 0.0], [10.0, 10.0]],
            },
            CANDIDATE,
            self.model,
        )

    def test_builds_candidate_tables_once(self):
        self.assertEqual([call[0] for call in self.model.table_calls], ["cd", "cd", "cn", "cp"])
        self.aero.coefficients(0.5, np.deg2rad(5.0), True)
        self.assertEqual(len(self.model.table_calls), 4)

    def test_passes_actual_nozzle_exit_without_clipping(self):
        candidate = {**CANDIDATE, "exit": 6.79}
        aero = Aero(
            {
                "aoa_schedule": [[0.0, 0.0], [10.0, 0.0]],
            },
            candidate,
            FakeDragModel(),
        )

        self.assertEqual(candidate["exit"], 6.79)
        self.assertEqual(aero.candidate["exit"], 6.79)

    def test_passes_actual_vehicle_length_without_clipping(self):
        for length in (309.0, 360.0, 420.0, 450.0):
            with self.subTest(length=length):
                candidate = {**CANDIDATE, "length": length}
                model = FakeDragModel()
                aero = Aero(
                    {"aoa_schedule": [[0.0, 0.0], [10.0, 0.0]]},
                    candidate,
                    model,
                )
                expected = length
                self.assertEqual(candidate["length"], length)
                self.assertEqual(aero.candidate["length"], expected)
                for call in model.table_calls:
                    passed = call[2] if call[0] == "cd" else call[1]
                    self.assertEqual(passed["length"], expected)
                distribution = aero.normal_distribution(0.5, np.deg2rad(5.0))
                self.assertAlmostEqual(distribution["x"][-1], expected * 0.0254)

    def test_invalid_geometry_is_rejected_not_clipped(self):
        for changes in ({"exit": 5.8}, {"length": 501.0}, {"length": 275.0}):
            with self.assertRaises(ValueError):
                Aero({"aoa_schedule": [[0, 0], [10, 0]]}, {**CANDIDATE, **changes}, FakeDragModel())

    def test_interpolates_tables_and_engine_state(self):
        alpha = np.deg2rad(5.0)
        cd_on, _, cn, cp = self.aero.coefficients(0.55, alpha, True)
        cd_off, _, _, _ = self.aero.coefficients(0.55, alpha, False)

        self.assertAlmostEqual(cd_on, 0.35)
        self.assertAlmostEqual(cd_off, 0.45)
        self.assertAlmostEqual(cn, 0.15)
        self.assertAlmostEqual(cp, 115.0 * 0.0254)

    def test_interpolates_aoa_schedule_in_radians(self):
        self.assertAlmostEqual(self.aero.aoa(5.0), np.deg2rad(5.0))

    def test_evaluate_returns_wind_and_body_axis_forces(self):
        kin = KinematicsState(5.0, 0.1, 0.0, 100.0, 0.0, np.deg2rad(5.0), 100.0, 1.0)
        atmosphere = AtmosState(288.0, 101325.0, 1.2, 1.8e-5, 340.0, 100.0, 0.55)

        result = self.aero.evaluate(kin, atmosphere, engine_on=True)
        scale = 100.0 * np.pi * (12.0 * 0.0254) ** 2 / 4.0

        self.assertAlmostEqual(result.Cd, 0.35)
        self.assertAlmostEqual(result.D, scale * result.Cd)
        self.assertAlmostEqual(result.A, scale * result.Ca)
        self.assertAlmostEqual(result.N, scale * result.Cn)

    def test_negative_alpha_reverses_cn_but_not_drag(self):
        positive = self.aero.coefficients(0.55, np.deg2rad(5.0), False)
        negative = self.aero.coefficients(0.55, np.deg2rad(-5.0), False)
        self.assertAlmostEqual(positive[0], negative[0])
        self.assertAlmostEqual(positive[1], negative[1])
        self.assertAlmostEqual(positive[2], -negative[2])

    def test_holds_first_table_value_below_mach_point_one(self):
        self.assertEqual(
            self.aero.coefficients(0.0, 0.0, True)[0],
            self.aero.coefficients(0.1, 0.0, True)[0],
        )

    def test_normal_distribution_uses_si_units(self):
        output = self.aero.normal_distribution(0.5, np.deg2rad(5.0))
        self.assertAlmostEqual(output["x"][-1], 360.0 * 0.0254)
        self.assertAlmostEqual(trapezoid(output["dcn_dx"], output["x"]), 0.1)

    def test_errors_outside_model_coordinates_and_schedule(self):
        with self.assertRaises(ValueError):
            self.aero.coefficients(1.1, 0.0, True)
        with self.assertRaises(ValueError):
            self.aero.coefficients(0.5, np.deg2rad(16.0), True)
        with self.assertRaises(ValueError):
            self.aero.aoa(11.0)


if __name__ == "__main__":
    unittest.main()
