import unittest

from Flight.FluidsDef import FluidsDef


class FluidsDefTests(unittest.TestCase):
    def test_near_critical_nitrogen_pt_state(self):
        state = FluidsDef.coolprop_state(
            fluid="Nitrogen",
            input_1="P",
            value_1=3_368_884.784,
            input_2="T",
            value_2=126.5,
        )

        self.assertAlmostEqual(state["T"], 126.5, places=3)
        self.assertGreater(state["rho"], 0.0)


if __name__ == "__main__":
    unittest.main()
