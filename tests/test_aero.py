import tempfile
import unittest
from pathlib import Path

import numpy as np

from Flight.flight_forces import Aero
from Flight.types import AtmosState, KinematicsState


DECK = """mach,aoa_deg,cd_engine_on,cd_engine_off
0.0,0.0,0.20,0.30
0.0,10.0,0.30,0.40
1.0,0.0,0.40,0.50
1.0,10.0,0.50,0.60
"""


class AeroTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.deck = Path(self.directory.name) / "cd.csv"
        self.deck.write_text(DECK)
        self.aero = Aero(
            {
                "reference_area": 2.0,
                "cd_table": self.deck,
                "aoa_schedule": [[0.0, 0.0], [10.0, 10.0]],
            }
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_interpolates_mach_aoa_and_engine_state(self):
        alpha = np.deg2rad(5.0)

        self.assertAlmostEqual(self.aero.cd(0.5, alpha, True), 0.35)
        self.assertAlmostEqual(self.aero.cd(0.5, alpha, False), 0.45)

    def test_interpolates_aoa_schedule_in_radians(self):
        self.assertAlmostEqual(self.aero.aoa(5.0), np.deg2rad(5.0))

    def test_evaluate_returns_cd_and_drag(self):
        kin = KinematicsState(
            t=5.0,
            dt=0.1,
            h=0.0,
            v=100.0,
            w=0.0,
            alpha=np.deg2rad(5.0),
            m=100.0,
            Ixx=1.0,
        )
        atmosphere = AtmosState(
            T=288.0,
            p=101325.0,
            rho=1.2,
            mu=1.8e-5,
            a=340.0,
            q=100.0,
            Ma=0.5,
        )

        result = self.aero.evaluate(kin, atmosphere, engine_on=True)

        self.assertAlmostEqual(result.Cd, 0.35)
        self.assertAlmostEqual(result.D, 70.0)

    def test_errors_outside_table_and_schedule(self):
        with self.assertRaises(ValueError):
            self.aero.cd(1.1, 0.0, True)
        with self.assertRaises(ValueError):
            self.aero.aoa(11.0)

    def test_rejects_incomplete_deck(self):
        incomplete = Path(self.directory.name) / "incomplete.csv"
        incomplete.write_text("\n".join(DECK.splitlines()[:-1]))

        with self.assertRaisesRegex(ValueError, "complete Mach/AoA grid"):
            Aero(
                {
                    "reference_area": 2.0,
                    "cd_table": incomplete,
                    "aoa_schedule": [[0.0, 0.0], [10.0, 10.0]],
                }
            )


if __name__ == "__main__":
    unittest.main()
