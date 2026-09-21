import unittest
from types import SimpleNamespace

import numpy as np

from Vehicle.utils.aoa import Cl_delta, aoa_from_flight_history, roll_pitch_yaw_upper_envelope
from Vehicle.utils.geometry import RocketAeroGeometry


class EnvelopeTests(unittest.TestCase):
    def calculate(self, **overrides):
        inputs = dict(time=np.linspace(0, 1, 11), qbar=np.ones(11), V=np.ones(11),
                      Ix=1, Iy=2, Iz=2, A=1, d=1, Cm_alpha=-8, Cm_q=-2,
                      Cl_delta=1, Cl_p=0, fin_cant=0, p0=0, R1_0=0, R2_0=0)
        inputs.update(overrides)
        return roll_pitch_yaw_upper_envelope(**inputs)

    def test_zero_and_symmetric_modes(self):
        result = self.calculate()
        np.testing.assert_array_equal(result['alpha_upper'], 0)
        np.testing.assert_array_equal(result['q_alpha_upper'], 0)
        np.testing.assert_array_equal(result['p'], 0)
        np.testing.assert_allclose(result['omega_1'], result['omega_2'])

    def test_decay_growth_and_left_endpoint(self):
        for damping in (-2, 2):
            result = self.calculate(Cm_q=damping, R1_0=.01, R2_0=.02)
            np.testing.assert_allclose(result['R1'], .01 * np.exp(damping / 8 * result['time']))
        damping = np.arange(11.)
        result = self.calculate(Cm_q=damping, R1_0=.01)
        expected = .01 * np.exp(np.r_[0, np.cumsum(damping[:-1] / 8 * .1)])
        np.testing.assert_allclose(result['R1'], expected)

    def test_trim_and_peaks(self):
        result = self.calculate(M_pitch_external=24, M_yaw_external=32)
        np.testing.assert_allclose(result['alpha_trim'], 5)
        self.assertEqual(result['peak_alpha_rad'], 5)
        self.assertEqual(result['peak_q_alpha'], 5)
        self.assertEqual(result['peak_q_alpha_time'], 0)

    def test_asymmetric_mass_and_roll_reuse(self):
        result = self.calculate(Iz=8, Cm_q=0)
        np.testing.assert_allclose(result['omega_1'], 1)
        np.testing.assert_allclose(result['omega_2'], 2)
        p = np.arange(11.)
        result = self.calculate(roll_rate=p, fin_cant=1, M_roll_external=100)
        np.testing.assert_array_equal(result['p'], p)
        result = self.calculate(fin_cant=.1)
        np.testing.assert_allclose(result['p'], .1 * result['time'])

    def test_real_unstable_modes(self):
        result = self.calculate(Cm_alpha=8, Cm_q=0, R1_0=.01)
        self.assertTrue(np.all(result['diagnostic_modes']))
        np.testing.assert_allclose(result['R1'], .01 * np.exp(2 * result['time']))

    def test_zero_speed_and_pressure(self):
        result = self.calculate(qbar=np.zeros(11), V=np.zeros(11), R1_0=.01)
        np.testing.assert_allclose(result['alpha_upper'], .01)
        np.testing.assert_array_equal(result['q_alpha_upper'], 0)

    def test_fin_geometry_and_recent_geometry_helpers(self):
        derivative = Cl_delta(4, .2, .1, .2, 2, .2)
        self.assertAlmostEqual(derivative, 10)
        self.assertAlmostEqual(Cl_delta(8, .2, .1, .2, 2, .2), 2 * derivative)
        self.assertAlmostEqual(Cl_delta(4, .4, .1, .2, 2, .2), 14)
        self.assertAlmostEqual(Cl_delta(4, .2, .2, .3, 2, .2), 14)
        geometry = RocketAeroGeometry(.2, .1, .1, 4, body_radius=.05)
        self.assertTrue(np.isfinite(geometry.roll_damping_interference_factor))
        self.assertIsNone(RocketAeroGeometry().body_radius)

    def test_adapter(self):
        rows = [dict(kinematics=SimpleNamespace(t=t, v=10),
                     atmosphere=SimpleNamespace(q=100),
                     mass_properties=dict(Ixx=1, Iyy=2)) for t in (0, .1)]
        geometry = RocketAeroGeometry(body_radius=.1, fin_cant=1)
        result = aoa_from_flight_history(rows, geometry, Iz=3, Cm_alpha=-1,
                                        Cm_q=-1, Cl_delta=1, Cl_p=0,
                                        p0=0, R1_0=.01, R2_0=.01)
        self.assertAlmostEqual(result['p'][1], 100 * geometry.reference_area *
                               geometry.diameter * np.deg2rad(1) * .1)

    def test_grid_refinement(self):
        peaks = []
        for n in (51, 101, 201):
            t = np.linspace(0, 1, n)
            result = self.calculate(time=t, qbar=1 + t, V=np.ones(n),
                                    Iy=2, Iz=3, fin_cant=.1, Cl_p=-1,
                                    R1_0=.01, R2_0=.02)
            peaks.append(result['peak_q_alpha'])
        self.assertLess(abs(peaks[2] - peaks[1]), abs(peaks[1] - peaks[0]))
        self.assertLess(abs(peaks[2] - peaks[1]) / peaks[2], .01)


if __name__ == '__main__':
    unittest.main()
