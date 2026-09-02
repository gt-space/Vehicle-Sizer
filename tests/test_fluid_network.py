import unittest
from dataclasses import dataclass

from CoolProp.CoolProp import PropsSI

from Flight.FluidNetwork import FluidNetwork


@dataclass(frozen=True)
class GasGeometry:
    volume: float
    internal_area: float

    @staticmethod
    def axial_mass(mass):
        return [mass]


@dataclass(frozen=True)
class TankGeometry:
    volume: float
    internal_area: float = 2.0

    def fill_state(self, liquid_volume):
        fraction = liquid_volume / self.volume
        return {
            "fill_height": fraction,
            "liquid_contact_area": self.internal_area * fraction,
            "ullage_contact_area": self.internal_area * (1.0 - fraction),
        }

    @staticmethod
    def axial_mass(liquid_volume, liquid_mass, ullage_mass):
        return [liquid_mass + ullage_mass]


def stored_state(fluid, pressure, temperature, volume):
    density = PropsSI("Dmass", "P", pressure, "T", temperature, fluid)
    internal_energy = PropsSI("Umass", "P", pressure, "T", temperature, fluid)
    mass = density * volume
    return mass, mass * internal_energy


class FakeCEA:
    def __init__(self):
        self.calls = 0

    def get_Cstar(self, Pc, MR):
        self.calls += 1
        return 2000.0

    @staticmethod
    def getFrozen_PambCf(Pamb, Pc, MR, eps, frozen):
        return 0.0, 1.5 + (100_000.0 - Pamb) / 1.0e6

    @staticmethod
    def get_Chamber_MolWt_gamma(Pc, MR, eps):
        return 24.0, 1.2

    @staticmethod
    def get_Temperatures(Pc, MR, eps):
        return (3200.0,)

    @staticmethod
    def get_Chamber_H(Pc, MR, eps):
        return 5.0e6


class FluidNetworkTests(unittest.TestCase):
    def test_runtime_network_does_not_require_design_circuits(self):
        network = FluidNetwork(
            nodes={
                "source": {"type": "boundary_pressure"},
                "sink": {"type": "boundary_pressure"},
            },
            branches={
                "feed": {
                    "type": "liquid_loss",
                    "fluid": "Water",
                    "from": "source",
                    "to": "sink",
                    "CdA": 1.0e-5,
                }
            },
        )

        self.assertFalse(hasattr(network, "circuits"))
        self.assertEqual(network.branches["feed"]["fluid"], "Water")

    def test_bang_bang_alone_applies_duty_cycle(self):
        gas_orifice = FluidNetwork._make_branch(
            "gas",
            {
                "type": "gas_orifice",
                "CdA": 2.0,
                "duty_cycle": 0.25,
            },
        )
        bang_bang = FluidNetwork._make_branch(
            "bang",
            {
                "type": "bang_bang",
                "CdA": 2.0,
                "duty_cycle": 0.25,
            },
        )

        self.assertEqual(gas_orifice.effective_cda(), 2.0)
        self.assertEqual(bang_bang.effective_cda(), 0.5)

    def test_combustion_node_recomputes_cea_during_network_update(self):
        cea = FakeCEA()
        oxidizer_cda = 3.0 / (2.0 * 1000.0 * 100_000.0) ** 0.5
        fuel_cda = 1.0 / (2.0 * 1000.0 * 100_000.0) ** 0.5
        network = FluidNetwork(
            nodes={
                "ox_source": {"type": "boundary_pressure"},
                "fuel_source": {"type": "boundary_pressure"},
                "chamber": {
                    "type": "comb_device",
                    "P0": 2.0e6,
                    "cea": cea,
                    "expansion_ratio": 5.0,
                    "oxidizer_fluid": "ox",
                    "fuel_fluid": "fuel",
                    "combustion_fluid": "combustion_gas",
                    "ambient_node": "ambient",
                },
                "ambient": {"type": "boundary_pressure"},
            },
            branches={
                "ox": {
                    "type": "liquid_loss",
                    "fluid": "ox",
                    "from": "ox_source",
                    "to": "chamber",
                    "CdA": oxidizer_cda,
                },
                "fuel": {
                    "type": "liquid_loss",
                    "fluid": "fuel",
                    "from": "fuel_source",
                    "to": "chamber",
                    "CdA": fuel_cda,
                },
                "nozzle": {
                    "type": "nozzle",
                    "fluid": "combustion_gas",
                    "from": "chamber",
                    "to": "ambient",
                    "At": 0.004,
                    "Cd": 1.0,
                },
            },
        )
        def source_state(fluid):
            return {
                "P": 2.1e6,
                "fluids": {fluid: {"rho": 1000.0, "h": 100_000.0}},
            }

        sea_level = network.update(
            bcs={
                "ox_source": source_state("ox"),
                "fuel_source": source_state("fuel"),
                "ambient": {"P": 100_000.0},
            }
        )
        calls_after_first_update = cea.calls
        altitude = network.update(
            bcs={
                "ox_source": source_state("ox"),
                "fuel_source": source_state("fuel"),
                "ambient": {"P": 50_000.0},
            }
        )

        self.assertAlmostEqual(sea_level["node"]["chamber"]["P"], 2.0e6, delta=1.0)
        self.assertAlmostEqual(sea_level["node"]["chamber"]["MR"], 3.0, places=6)
        self.assertAlmostEqual(sea_level["mdot"]["nozzle"], 4.0, places=6)
        self.assertIn("combustion_gas", sea_level["node"]["chamber"]["fluids"])
        self.assertGreater(cea.calls, calls_after_first_update)
        self.assertGreater(
            altitude["node"]["chamber"]["Cf"],
            sea_level["node"]["chamber"]["Cf"],
        )

    def test_nozzle_uses_cstar_mass_flow(self):
        network = FluidNetwork(
            nodes={
                "chamber": {"type": "boundary_pressure"},
                "ambient": {"type": "boundary_pressure"},
            },
            branches={
                "nozzle": {
                    "type": "nozzle",
                    "fluid": "combustion_gas",
                    "from": "chamber",
                    "to": "ambient",
                    "At": 0.005,
                    "Cd": 0.8,
                }
            },
        )

        result = network.update(
            bcs={
                "chamber": {
                    "P": 2.0e6,
                    "cstar": 2000.0,
                    "fluids": {"combustion_gas": {"h": 5.0e6}},
                },
                "ambient": {"P": 100_000.0},
            },
        )

        self.assertAlmostEqual(result["mdot"]["nozzle"], 4.0)

    def test_algebraic_node_uses_the_same_update_path(self):
        network = FluidNetwork(
            nodes={
                "source": {"type": "boundary_pressure"},
                "junction": {
                    "type": "liquid_volume",
                    "P0": 200_000.0,
                },
                "sink": {"type": "boundary_pressure"},
            },
            branches={
                "in": {
                    "type": "liquid_loss",
                    "fluid": "water",
                    "from": "source",
                    "to": "junction",
                    "CdA": 1.0e-5,
                },
                "out": {
                    "type": "liquid_loss",
                    "fluid": "water",
                    "from": "junction",
                    "to": "sink",
                    "CdA": 1.0e-5,
                },
            },
        )

        result = network.update(
            bcs={
                "source": {
                    "P": 300_000.0,
                    "fluids": {"water": {"rho": 1000.0, "h": 100_000.0}},
                },
                "sink": {"P": 100_000.0},
            },
        )

        self.assertAlmostEqual(result["node"]["junction"]["P"], 200_000.0)
        self.assertAlmostEqual(result["mdot"]["in"], result["mdot"]["out"])

    def test_gas_volume_propagates_and_commits(self):
        initial_mass, initial_energy = stored_state(
            "Nitrogen", 200_000.0, 300.0, 1.0
        )
        network = FluidNetwork(
            nodes={
                "tank": {
                    "type": "gas_volume",
                    "fluid": "Nitrogen",
                    "steady": False,
                    "geometry": GasGeometry(1.0, 2.0),
                    "state0": {"m": initial_mass, "U": initial_energy},
                },
                "ambient": {"type": "boundary_pressure"},
            },
            branches={
                "vent": {
                    "type": "gas_orifice",
                    "fluid": "Nitrogen",
                    "from": "tank",
                    "to": "ambient",
                    "CdA": 1.0e-5,
                }
            },
        )

        result = network.update(
            dt=0.1,
            bcs={"ambient": {"P": 100_000.0}},
        )

        self.assertLess(result["td_state"]["tank"]["m"], initial_mass)
        self.assertGreater(result["mdot"]["vent"], 0.0)
        self.assertIn("Nitrogen", result["node"]["tank"]["fluids"])

    def test_propellant_tank_updates_liquid_state(self):
        m_liq, U_liq = stored_state("Water", 200_000.0, 300.0, 1.0)
        m_ull, U_ull = stored_state("Nitrogen", 200_000.0, 300.0, 0.1)
        network = FluidNetwork(
            nodes={
                "tank": {
                    "type": "propellant_tank",
                    "steady": False,
                    "geometry": TankGeometry(1.1),
                    "P0": 200_000.0,
                    "state0": {
                        "m_liq": m_liq,
                        "U_liq": U_liq,
                        "m_ull": m_ull,
                        "U_ull": U_ull,
                    },
                    "liquid_fluid": "Water",
                    "gas_fluid": "Nitrogen",
                },
                "ambient": {"type": "boundary_pressure"},
            },
            branches={
                "out": {
                    "type": "liquid_loss",
                    "fluid": "Water",
                    "from": "tank",
                    "to": "ambient",
                    "CdA": 1.0e-5,
                }
            },
        )

        result = network.update(
            dt=0.01,
            bcs={"ambient": {"P": 100_000.0}},
        )

        self.assertLess(result["td_state"]["tank"]["m_liq"], m_liq)
        self.assertAlmostEqual(result["td_state"]["tank"]["m_ull"], m_ull)
        self.assertEqual(
            set(result["node"]["tank"]["fluids"]), {"Water", "Nitrogen"}
        )

    def test_propellant_tank_splits_node_heat_flux_using_current_fill(self):
        m_liq, initial_liquid_energy = stored_state(
            "Water", 200_000.0, 300.0, 1.0
        )
        m_ull, initial_ullage_energy = stored_state(
            "Nitrogen", 200_000.0, 300.0, 0.1
        )
        network = FluidNetwork(
            nodes={
                "tank": {
                    "type": "propellant_tank",
                    "geometry": TankGeometry(1.1),
                    "P0": 200_000.0,
                    "state0": {
                        "m_liq": m_liq,
                        "U_liq": initial_liquid_energy,
                        "m_ull": m_ull,
                        "U_ull": initial_ullage_energy,
                    },
                    "liquid_fluid": "Water",
                    "gas_fluid": "Nitrogen",
                },
            },
            branches={},
        )

        result = network.update(dt=0.1, heat_flux={"tank": 100_000.0})

        liquid_volume = result["node"]["tank"]["fluids"]["Water"]["V"]
        ullage_volume = result["node"]["tank"]["fluids"]["Nitrogen"]["V"]
        self.assertAlmostEqual(liquid_volume + ullage_volume, 1.1)
        self.assertGreater(result["td_state"]["tank"]["U_liq"], initial_liquid_energy)
        self.assertGreater(result["td_state"]["tank"]["U_ull"], initial_ullage_energy)
        self.assertAlmostEqual(
            result["td_state"]["tank"]["U_liq"]
            + result["td_state"]["tank"]["U_ull"]
            - initial_liquid_energy
            - initial_ullage_energy,
            20_000.0,
            delta=1.0,
        )

    def test_fixed_head_pump_is_an_algebraic_branch(self):
        network = FluidNetwork(
            nodes={
                "source": {"type": "boundary_pressure"},
                "pump_out": {
                    "type": "liquid_volume",
                    "P0": 300_000.0,
                },
                "sink": {"type": "boundary_pressure"},
            },
            branches={
                "pump": {
                    "type": "pump",
                    "fluid": "water",
                    "from": "source",
                    "to": "pump_out",
                    "dP": 200_000.0,
                    "CdA": None,
                },
                "loss": {
                    "type": "liquid_loss",
                    "fluid": "water",
                    "from": "pump_out",
                    "to": "sink",
                    "CdA": 1.0 / (2.0 * 1000.0 * 200_000.0) ** 0.5,
                },
            },
        )

        result = network.update(
            bcs={
                "source": {
                    "P": 100_000.0,
                    "fluids": {"water": {"rho": 1000.0, "h": 100_000.0}},
                },
                "sink": {"P": 100_000.0},
            },
        )

        self.assertAlmostEqual(result["node"]["pump_out"]["P"], 300_000.0)
        self.assertAlmostEqual(result["mdot"]["pump"], 1.0)


if __name__ == "__main__":
    unittest.main()
