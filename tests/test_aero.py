import unittest

import numpy as np

from Flight.flight_forces import Aero
from Flight.types import AtmosState, KinematicsState


class FakeModel:
    def __init__(self):
        self.mach = np.array([0.1, 1.0])
        self.alpha = np.array([0.0, 10.0])

    @staticmethod
    def check(candidate):
        if candidate.get("omld", 0.0) <= 0.0:
            raise ValueError("invalid candidate")

    @staticmethod
    def cd_table(candidate, nose, finish, power_on=False):
        del candidate, nose, finish
        if power_on:
            return np.array([[0.20, 0.30], [0.40, 0.50]])
        return np.array([[0.30, 0.40], [0.50, 0.60]])

    @staticmethod
    def cn_table(candidate, nose):
        del candidate, nose
        return np.array([[0.0, 0.20], [0.0, 0.40]])

    @staticmethod
    def cp_table(candidate, nose, fins_on_boattail=True):
        del candidate, nose, fins_on_boattail
        return np.array([[40.0, 42.0], [44.0, 46.0]])


class AeroTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.aero = Aero(
            {
                "nose": "vonkarman",
                "finish": "10um",
            },
            {"omld": 10.0},
            self.model,
        )

    def test_interpolates_mach_alpha_and_engine_state(self):
        alpha = np.deg2rad(5.0)

        cd_on, ca_on, cn_on, cp_on = self.aero.coefficients(0.55, alpha, True)
        cd_off, ca_off, cn_off, cp_off = self.aero.coefficients(0.55, alpha, False)

        self.assertAlmostEqual(cd_on, 0.35)
        self.assertAlmostEqual(cd_off, 0.45)
        self.assertAlmostEqual(cn_on, 0.15)
        self.assertAlmostEqual(cn_off, 0.15)
        self.assertAlmostEqual(cp_on, 43.0 * 0.0254)
        self.assertAlmostEqual(cp_off, cp_on)
        self.assertLess(ca_on, ca_off)

    def test_negative_alpha_reverses_normal_force_only(self):
        positive = self.aero.coefficients(0.55, np.deg2rad(5.0), True)
        negative = self.aero.coefficients(0.55, np.deg2rad(-5.0), True)

        self.assertAlmostEqual(negative[0], positive[0])
        self.assertAlmostEqual(negative[1], positive[1])
        self.assertAlmostEqual(negative[2], -positive[2])
        self.assertAlmostEqual(negative[3], positive[3])

    def test_evaluate_returns_current_3dof_aero_output(self):
        kin = KinematicsState(
            t=5.0,
            dt=0.1,
            x=0.0,
            h=0.0,
            vx=100.0,
            vz=0.0,
            theta=np.deg2rad(5.0),
            q=0.0,
            alpha=np.deg2rad(5.0),
            m=100.0,
            Iyy=1.0,
        )
        atmosphere = AtmosState(
            T=288.0,
            p=101325.0,
            rho=1.2,
            mu=1.8e-5,
            a=340.0,
            q=100.0,
            Ma=0.55,
        )

        result = self.aero.evaluate(kin, atmosphere, engine_on=True)
        scale = atmosphere.q * self.aero.reference_area

        self.assertAlmostEqual(result.Cd, 0.35)
        self.assertAlmostEqual(result.D, scale * result.Cd)
        self.assertAlmostEqual(result.A, scale * result.Ca)
        self.assertAlmostEqual(result.N, scale * result.Cn)
        self.assertTrue(np.isfinite(result.cp))

    def test_rejects_coordinates_outside_aero_table(self):
        with self.assertRaises(ValueError):
            self.aero.coefficients(1.1, 0.0, True)
        with self.assertRaises(ValueError):
            self.aero.coefficients(0.5, np.deg2rad(11.0), True)

    def test_aoa_schedule_is_not_required_for_3dof_aero(self):
        with self.assertRaises(RuntimeError):
            self.aero.aoa(5.0)

    def test_legacy_schedule_helper_still_returns_radians_when_configured(self):
        aero = Aero(
            {
                "aoa_schedule": [[0.0, 0.0], [10.0, 10.0]],
                "nose": "vonkarman",
                "finish": "10um",
            },
            {"omld": 10.0},
            self.model,
        )
        self.assertAlmostEqual(aero.aoa(5.0), np.deg2rad(5.0))
        with self.assertRaises(ValueError):
            aero.aoa(11.0)


if __name__ == "__main__":
    unittest.main()
