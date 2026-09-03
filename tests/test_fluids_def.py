import unittest

from Flight.FluidsDef import FluidsDef


class FluidsDefTests(unittest.TestCase):
    def test_near_critical_nitrogen_pu_state(self):
        state = FluidsDef.coolprop_state_pu(
            fluid="Nitrogen",
            pressure=3_368_884.784,
            internal_energy=231_756.3907,
            phase="gas",
        )

        self.assertAlmostEqual(state["T"], 321.0275, places=3)
        self.assertGreater(state["rho"], 0.0)


if __name__ == "__main__":
    unittest.main()
