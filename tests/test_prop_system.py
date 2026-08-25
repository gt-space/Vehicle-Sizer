import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from CoolProp.CoolProp import PropsSI

from Flight.PropSystem import PropSystem


@dataclass(frozen=True)
class GasGeometry:
    volume: float
    internal_area: float = 1.0


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


class GeometrySource:
    def __init__(self, geometry):
        self.geometry = geometry

    def get_fluid_geometry(self):
        return self.geometry


class FakeCEA:
    @staticmethod
    def get_eps_at_PcOvPe(Pc, MR, PcOvPe):
        return 5.0

    @staticmethod
    def get_Cstar(Pc, MR):
        return 1500.0

    @staticmethod
    def getFrozen_PambCf(Pamb, Pc, MR, eps, frozen):
        return 0.0, 1.5

    @staticmethod
    def get_Chamber_MolWt_gamma(Pc, MR, eps):
        return 24.0, 1.2

    @staticmethod
    def get_Temperatures(Pc, MR, eps):
        return (3000.0,)

    @staticmethod
    def get_Chamber_H(Pc, MR, eps):
        return 5.0e6


class PropSystemSizingTests(unittest.TestCase):
    def _config(self):
        return {
            "prop_system": {
                "press_model": "pressure_fed",
                "duty_cyl": 0.5,
                "Pc_target": 2.0e6,
                "MR_target": 3.0,
                "thrust_target": 12_000.0,
                "fuel_inj_stiffness": 0.2,
                "ox_inj_stiffness": 0.2,
                "fuel_tank_inj_dp": 200_000.0,
                "ox_tank_inj_dp": 200_000.0,
                "expansion_ratio": 5.0,
                "nozzle_cd": 1.0,
                "state0": {
                    "press_tank": {"m": 5.0, "U": 1.0e6},
                    "ox_tank": {
                        "fluid": "Oxygen",
                        "P": 2.6e6,
                        "T": 100.0,
                        "m_liq": 90.0,
                        "U_liq": 1.0e7,
                        "m_ull": 0.1,
                        "U_ull": 20_000.0,
                    },
                    "fuel_tank": {
                        "fluid": "n-Dodecane",
                        "P": 2.6e6,
                        "T": 300.0,
                        "m_liq": 30.0,
                        "U_liq": 1.0e7,
                        "m_ull": 0.1,
                        "U_ull": 20_000.0,
                    },
                },
            },
            "engine": {
                "oxidizer": "LOX",
                "fuel": "RP-1",
                "cstar_efficiency": 1.0,
                "cf_efficiency": 1.0,
                "exit_pressure": 100_000.0,
            },
            "press_tank": {
                "pressurant": "Nitrogen",
                "design_pressure": 30.0e6,
                "start_temp": 300.0,
                "min_temp": 220.0,
                "collapse_factor": 1.2,
            },
        }

    @staticmethod
    def _tanks(tank_pressure=2.6e6):
        ox_density = PropsSI("Dmass", "P", tank_pressure, "T", 100.0, "Oxygen")
        fuel_density = PropsSI(
            "Dmass", "P", tank_pressure, "T", 300.0, "n-Dodecane"
        )
        return (
            GeometrySource(TankGeometry(90.0 / ox_density * 1.1)),
            GeometrySource(TankGeometry(30.0 / fuel_density * 1.1)),
            GeometrySource(GasGeometry(0.05)),
        )

    @staticmethod
    def _make_system(config, ox_tank, fuel_tank, press_tank):
        with patch("Flight.PropSystem.CEA_Obj", return_value=FakeCEA()):
            return PropSystem(
                config,
                {
                    "ox_tank": ox_tank,
                    "fuel_tank": fuel_tank,
                    "press_tank": press_tank,
                },
            )

    def test_pressure_ladder_sizes_engine_and_branches(self):
        system = self._make_system(self._config(), *self._tanks())

        self.assertAlmostEqual(system.throat_area, 0.004)
        self.assertAlmostEqual(system.mdot_total, 16.0 / 3.0)
        self.assertAlmostEqual(system.mdot_ox, 4.0)
        self.assertAlmostEqual(system.mdot_fuel, 4.0 / 3.0)
        self.assertEqual(system.network.node_definitions["ox_inj_in"]["P0"], 2.4e6)
        self.assertGreater(system.network.branches["OX_INJ"]["CdA"], 0.0)
        self.assertGreater(system.network.branches["OX_BANGBANG"]["CdA"], 0.0)
        self.assertEqual(system.network.branches["OX_BANGBANG"]["type"], "bang_bang")
        self.assertNotIn("displaced_fluid", system.network.branches["OX_BANGBANG"])
        self.assertEqual(system.network.branches["NOZZLE"]["At"], system.throat_area)
        self.assertEqual(
            system.network.node_definitions["thrust_chamber"]["design_state"]["cstar"],
            system.cstar,
        )
        self.assertEqual(system.network.branches["NOZZLE"]["Cd"], 1.0)

    def test_pressure_fed_update_returns_thrust(self):
        config = self._config()
        tank_pressure = 2.6e6
        gas_temperature = 260.0
        ox_density = PropsSI("Dmass", "P", tank_pressure, "T", 100.0, "Oxygen")
        fuel_density = PropsSI(
            "Dmass", "P", tank_pressure, "T", 300.0, "n-Dodecane"
        )
        ox_volume = (90.0 / ox_density) * 1.1
        fuel_volume = (30.0 / fuel_density) * 1.1
        press_volume = 0.05

        def stored_state(fluid, pressure, temperature, volume):
            density = PropsSI("Dmass", "P", pressure, "T", temperature, fluid)
            internal_energy = PropsSI(
                "Umass", "P", pressure, "T", temperature, fluid
            )
            mass = density * volume
            return mass, mass * internal_energy

        ox_gas_mass, ox_gas_energy = stored_state(
            "Nitrogen", tank_pressure, gas_temperature, ox_volume - 90.0 / ox_density
        )
        fuel_gas_mass, fuel_gas_energy = stored_state(
            "Nitrogen",
            tank_pressure,
            gas_temperature,
            fuel_volume - 30.0 / fuel_density,
        )
        press_mass, press_energy = stored_state(
            "Nitrogen", 30.0e6, 300.0, press_volume
        )
        ox_mass, ox_energy = stored_state(
            "Oxygen", tank_pressure, 100.0, 90.0 / ox_density
        )
        fuel_mass, fuel_energy = stored_state(
            "n-Dodecane", tank_pressure, 300.0, 30.0 / fuel_density
        )
        config["prop_system"]["state0"] = {
            "press_tank": {"m": press_mass, "U": press_energy},
            "ox_tank": {
                "fluid": "Oxygen",
                "P": tank_pressure,
                "T": 100.0,
                "m_liq": ox_mass,
                "U_liq": ox_energy,
                "m_ull": ox_gas_mass,
                "U_ull": ox_gas_energy,
            },
            "fuel_tank": {
                "fluid": "n-Dodecane",
                "P": tank_pressure,
                "T": 300.0,
                "m_liq": fuel_mass,
                "U_liq": fuel_energy,
                "m_ull": fuel_gas_mass,
                "U_ull": fuel_gas_energy,
            },
        }
        system = self._make_system(
            config,
            GeometrySource(TankGeometry(ox_volume)),
            GeometrySource(TankGeometry(fuel_volume)),
            GeometrySource(GasGeometry(press_volume)),
        )

        result = system.update(
            dt=0.001,
            atm=SimpleNamespace(p=100_000.0),
            heat_flux={},
        )

        self.assertAlmostEqual(result["propulsion"]["MR"], 3.0, places=3)
        self.assertAlmostEqual(result["propulsion"]["thrust"], 12_000.0, delta=20.0)
        self.assertAlmostEqual(
            result["propulsion"]["mdot_nozzle"],
            result["propulsion"]["mdot_ox"] + result["propulsion"]["mdot_fuel"],
            places=7,
        )


if __name__ == "__main__":
    unittest.main()
