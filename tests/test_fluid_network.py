import unittest
from copy import deepcopy
from dataclasses import dataclass

from CoolProp.CoolProp import PropsSI

from Flight.FluidBranch import (
    BangBangValveComponent,
    CompressibleLossModel,
    FluidBranch,
    IncompressibleLossModel,
    TwinPathNozzleModel,
)
from Flight.FluidNetwork import FluidNetwork, NetworkState
from Flight.FluidNode import (
    CombustionModel,
    CombustorComponent,
    FluidNode,
    JunctionModel,
    TwinPathJunctionModel,
)
from Flight.FluidState import BranchState, FluidState, NodeState
from FluidProperties.PropertyModels import CEAPropertySource, CoolPropPropertySource


def branch_flow(branch_id, incidence, state):
    del branch_id
    return incidence, BranchState(
        enabled=state["enabled"],
        flows={
            name: {
                "mdot": values["mdot"],
                "direction": 1,
                "fluid": FluidState.from_dict(name, values),
            }
            for name, values in state["components"].items()
        },
    )


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
    def test_components_own_models_while_plain_elements_use_models_directly(self):
        junction = FluidNetwork._make_node(
            "junction", {"model": "junction", "P0": 100_000.0}
        )
        combustor = FluidNetwork._make_node(
            "chamber", {"component": "combustor", "P0": 2.0e6}
        )
        loss = FluidNetwork._make_branch(
            "loss",
            {"model": "compressible_loss", "fluid": "gas", "CdA": 1.0},
        )
        valve = FluidNetwork._make_branch(
            "valve",
            {
                "component": "bang_bang_valve",
                "fluid": "gas",
                "CdA": 1.0,
                "duty_cycle": 0.5,
                "pressure_band": 10_000.0,
                "target_pressure": 200_000.0,
                "initially_open": True,
            },
        )

        self.assertIs(type(junction), FluidNode)
        self.assertIsInstance(junction.model, JunctionModel)
        self.assertIsInstance(combustor, CombustorComponent)
        self.assertIsInstance(combustor.model, CombustionModel)
        self.assertIs(type(loss), FluidBranch)
        self.assertIsInstance(loss.model, CompressibleLossModel)
        self.assertIsInstance(valve, BangBangValveComponent)
        self.assertIsInstance(valve.model, CompressibleLossModel)

    def test_runtime_network_does_not_require_design_circuits(self):
        network = FluidNetwork(
            nodes={
                "source": {"model": "boundary"},
                "sink": {"model": "boundary"},
            },
            branches={
                "feed": {
                    "model": "incompressible_loss",
                    "fluid": "Water",
                    "from": "source",
                    "to": "sink",
                    "CdA": 1.0e-5,
                }
            },
        )

        self.assertFalse(hasattr(network, "circuits"))
        self.assertEqual(network.branches["feed"].fluid, "Water")

    def test_bang_bang_uses_full_sized_area_when_open(self):
        gas_orifice = FluidNetwork._make_branch(
            "gas",
            {
                "model": "compressible_loss",
                "fluid": "gas",
                "CdA": 2.0,
                "duty_cycle": 0.25,
            },
        )
        bang_bang = FluidNetwork._make_branch(
            "bang",
            {
                "component": "bang_bang_valve",
                "fluid": "gas",
                "CdA": 2.0,
                "duty_cycle": 0.25,
                "pressure_band": 10_000.0,
                "target_pressure": 200_000.0,
                "initially_open": True,
            },
        )

        self.assertEqual(gas_orifice.effective_cda(), 2.0)
        self.assertEqual(bang_bang.effective_cda(), 2.0)

    def test_bang_bang_retains_state_inside_pressure_band(self):
        valve = FluidNetwork._make_branch(
            "bang",
            {
                "component": "bang_bang_valve",
                "fluid": "gas",
                "CdA": 2.0,
                "duty_cycle": 0.25,
                "pressure_band": 10_000.0,
                "target_pressure": 200_000.0,
                "initially_open": True,
            },
        )

        self.assertFalse(valve.update_control(205_000.0))
        self.assertTrue(valve.is_open)
        valve.state["mdot"] = 0.4
        self.assertTrue(valve.update_control(210_000.0))
        self.assertFalse(valve.is_open)
        self.assertEqual(valve.solver_state(), {})
        self.assertFalse(valve.update_control(195_000.0))
        self.assertFalse(valve.is_open)
        self.assertTrue(valve.update_control(190_000.0))
        self.assertTrue(valve.is_open)
        self.assertEqual(valve.state["mdot"], 0.4)

    def test_combustion_node_recomputes_cea_during_network_update(self):
        cea = FakeCEA()
        oxidizer_cda = 3.0 / (2.0 * 1000.0 * 100_000.0) ** 0.5
        fuel_cda = 1.0 / (2.0 * 1000.0 * 100_000.0) ** 0.5
        network = FluidNetwork(
            nodes={
                "ox_source": {"model": "boundary"},
                "fuel_source": {"model": "boundary"},
                "chamber": {
                    "component": "combustor",
                    "P0": 2.0e6,
                    "expansion_ratio": 5.0,
                    "oxidizer_fluid": "ox",
                    "fuel_fluid": "fuel",
                    "combustion_fluid": "combustion_gas",
                    "ambient_node": "ambient",
                    "cstar_efficiency": 1.0,
                    "cf_efficiency": 1.0,
                },
                "ambient": {"model": "boundary"},
            },
            branches={
                "ox": {
                    "model": "incompressible_loss",
                    "fluid": "ox",
                    "from": "ox_source",
                    "to": "chamber",
                    "CdA": oxidizer_cda,
                },
                "fuel": {
                    "model": "incompressible_loss",
                    "fluid": "fuel",
                    "from": "fuel_source",
                    "to": "chamber",
                    "CdA": fuel_cda,
                },
                "nozzle": {
                    "component": "nozzle",
                    "fluid": "combustion_gas",
                    "from": "chamber",
                    "to": "ambient",
                    "At": 0.004,
                    "Cd": 1.0,
                },
            },
            combustion_properties=CEAPropertySource(cea),
        )
        def source_state(fluid):
            return {
                "P": 2.1e6,
                "fluids": {
                    fluid: {
                        "phase": "liquid",
                        "rho": 1000.0,
                        "h": 100_000.0,
                    }
                },
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
                "chamber": {"model": "boundary"},
                "ambient": {"model": "boundary"},
            },
            branches={
                "nozzle": {
                    "component": "nozzle",
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
                    "fluids": {
                        "combustion_gas": {"phase": "gas", "h": 5.0e6}
                    },
                },
                "ambient": {"P": 100_000.0},
            },
        )

        self.assertAlmostEqual(result["mdot"]["nozzle"], 4.0)

    def test_algebraic_node_uses_the_same_update_path(self):
        network = FluidNetwork(
            nodes={
                "source": {"model": "boundary"},
                "junction": {
                    "model": "junction",
                    "P0": 200_000.0,
                },
                "sink": {"model": "boundary"},
            },
            branches={
                "in": {
                    "model": "incompressible_loss",
                    "fluid": "water",
                    "from": "source",
                    "to": "junction",
                    "CdA": 1.0e-5,
                },
                "out": {
                    "model": "incompressible_loss",
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
                    "fluids": {
                        "water": {
                            "phase": "liquid",
                            "rho": 1000.0,
                            "h": 100_000.0,
                        }
                    },
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
                    "component": "pressurant_tank",
                    "fluid": "Nitrogen",
                    "steady": False,
                    "geometry": GasGeometry(1.0, 2.0),
                    "state0": {
                        "P": 200_000.0,
                        "T": 300.0,
                        "m": initial_mass,
                        "U": initial_energy,
                    },
                },
                "ambient": {"model": "boundary"},
            },
            branches={
                "vent": {
                    "model": "compressible_loss",
                    "fluid": "Nitrogen",
                    "from": "tank",
                    "to": "ambient",
                    "CdA": 1.0e-5,
                }
            },
            fluid_properties=CoolPropPropertySource(),
        )

        result = network.update(
            dt=0.1,
            bcs={"ambient": {"P": 100_000.0}},
        )

        self.assertLess(result["td_state"]["tank"]["m"], initial_mass)
        self.assertGreater(result["mdot"]["vent"], 0.0)
        self.assertIn("Nitrogen", result["node"]["tank"]["fluids"])

    def test_noncommitted_solve_does_not_change_network_state(self):
        pressure = 200_000.0
        temperature = 300.0
        properties = CoolPropPropertySource()
        gas = properties.state_pt("Nitrogen", pressure, temperature)
        network = FluidNetwork(
            nodes={
                "tank": {
                    "component": "pressurant_tank",
                    "fluid": "Nitrogen",
                    "geometry": GasGeometry(1.0, 2.0),
                    "state0": {
                        "P": pressure,
                        "T": temperature,
                        "m": gas.rho,
                        "U": gas.rho * gas.u,
                    },
                },
                "ambient": {"model": "boundary"},
            },
            branches={
                "vent": {
                    "model": "compressible_loss",
                    "fluid": "Nitrogen",
                    "from": "tank",
                    "to": "ambient",
                    "CdA": 1.0e-5,
                }
            },
            fluid_properties=properties,
        )
        boundaries = {"ambient": {"P": 100_000.0}}
        network.update(bcs=boundaries, commit=True)
        stored_node = deepcopy(network.nodes["tank"].state)
        stored_branch = deepcopy(network.branches["vent"].state)
        stored_output = deepcopy(network.state)

        candidate = network.update(dt=0.1, bcs=boundaries, commit=False)

        self.assertLess(candidate["td_state"]["tank"]["m"], stored_node["m"])
        self.assertEqual(network.nodes["tank"].state, stored_node)
        self.assertEqual(network.branches["vent"].state, stored_branch)
        self.assertEqual(network.state.nodes, stored_output.nodes)

    def test_event_step_handles_simultaneous_and_sequential_events(self):
        fired = []

        class TimedEvent:
            def __init__(self, event_id, event_time):
                self.id = event_id
                self.event_time = event_time
                self.active = True

            def event_values(self, node_state):
                if not self.active:
                    return {}
                return {"switch": self.event_time - node_state["clock"]["t"]}

            def apply_event(self, name):
                self.active = False
                fired.append(self.id)
                return True

        network = FluidNetwork.__new__(FluidNetwork)
        network.nodes = {}
        network.branches = {
            "first": TimedEvent("first", 0.25),
            "same_time": TimedEvent("same_time", 0.25),
            "later": TimedEvent("later", 0.75),
        }
        network.state = NetworkState(nodes={"clock": NodeState({"t": 0.0})})

        def solve(dt, bcs, heat_flux, commit=False):
            time = network.state.node["clock"]["t"] + (dt or 0.0)
            candidate = NetworkState(nodes={"clock": NodeState({"t": time})})
            return {
                "success": True,
                "message": "test candidate",
                "state": candidate,
                "node": {"clock": {"t": time}},
                "branch": {},
                "td_state": {},
                "mdot": {},
            }

        def commit(candidate, bcs):
            network.state = candidate

        network._solve = solve
        network._commit_candidate = commit

        result = network.update(dt=1.0)

        self.assertEqual(fired, ["first", "same_time", "later"])
        self.assertAlmostEqual(result["node"]["clock"]["t"], 1.0)

    def test_single_species_volume_switches_to_saturated_two_phase(self):
        properties = CoolPropPropertySource()
        pressure = 500_000.0
        saturation = properties.saturation_at_p("Nitrogen", pressure)
        temperature = saturation.T + 0.1
        gas = properties.state_pt("Nitrogen", pressure, temperature)
        mass = gas.rho
        network = FluidNetwork(
            nodes={
                "tank": {
                    "component": "pressurant_tank",
                    "fluid": "Nitrogen",
                    "geometry": GasGeometry(1.0, 2.0),
                    "state0": {
                        "P": pressure,
                        "T": temperature,
                        "m": mass,
                        "U": mass * gas.u,
                    },
                }
            },
            branches={},
            fluid_properties=properties,
        )
        network.update(commit=True)
        tank = network.nodes["tank"]
        target_quality = 0.4
        specific_volume = (
            (1.0 - target_quality) / saturation.liquid.rho
            + target_quality / saturation.vapor.rho
        )
        mass = 1.0 / specific_volume
        specific_energy = (
            (1.0 - target_quality) * saturation.liquid.u
            + target_quality * saturation.vapor.u
        )
        tank.state.update(
            {
                "T": saturation.T,
                "m": mass,
                "U": mass * specific_energy,
            }
        )
        network.state.node["tank"].update(tank.state)

        self.assertTrue(network._apply_transitions())
        result = network.update(dt=0.01)

        self.assertIn("quality", result["td_state"]["tank"])
        self.assertAlmostEqual(
            result["td_state"]["tank"]["quality"], target_quality
        )
        self.assertEqual(
            result["node"]["tank"]["fluids"]["Nitrogen"]["phase"], "gas"
        )

    def test_propellant_tank_updates_liquid_state(self):
        m_liq, U_liq = stored_state("Water", 200_000.0, 300.0, 1.0)
        m_ull, U_ull = stored_state("Nitrogen", 200_000.0, 300.0, 0.1)
        network = FluidNetwork(
            nodes={
                "tank": {
                    "component": "propellant_tank",
                    "steady": False,
                    "geometry": TankGeometry(1.1),
                    "P0": 200_000.0,
                    "state0": {
                        "P": 200_000.0,
                        "T": 300.0,
                        "gas_T": 300.0,
                        "m_liq": m_liq,
                        "U_liq": U_liq,
                        "m_ull": m_ull,
                        "U_ull": U_ull,
                    },
                    "liquid_fluid": "Water",
                    "gas_fluid": "Nitrogen",
                },
                "ambient": {"model": "boundary"},
            },
            branches={
                "out": {
                    "model": "incompressible_loss",
                    "fluid": "Water",
                    "from": "tank",
                    "to": "ambient",
                    "CdA": 1.0e-5,
                }
            },
            fluid_properties=CoolPropPropertySource(),
        )

        result = network.update(
            dt=0.01,
            bcs={"ambient": {"P": 100_000.0}},
            axial_specific_force=20.0,
        )

        self.assertLess(result["td_state"]["tank"]["m_liq"], m_liq)
        self.assertAlmostEqual(result["td_state"]["tank"]["m_ull"], m_ull)
        self.assertEqual(
            set(result["node"]["tank"]["fluids"]), {"Water", "Nitrogen"}
        )
        tank = result["node"]["tank"]
        expected_outlet_pressure = (
            tank["P"]
            + tank["fluids"]["Water"]["rho"]
            * 20.0
            * tank["fill_height"]
        )
        self.assertAlmostEqual(
            tank["port_pressure"]["liquid"], expected_outlet_pressure
        )
        self.assertAlmostEqual(tank["port_pressure"]["ullage"], tank["P"])
        self.assertAlmostEqual(
            result["branch"]["out"]["dP"],
            expected_outlet_pressure - 100_000.0,
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
                    "component": "propellant_tank",
                    "geometry": TankGeometry(1.1),
                    "P0": 200_000.0,
                    "state0": {
                        "P": 200_000.0,
                        "T": 300.0,
                        "gas_T": 300.0,
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
            fluid_properties=CoolPropPropertySource(),
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

    def test_tank_dryout_does_not_close_flow_path(self):
        pressure = 300_000.0
        liquid_volume = 1.0e-6
        ullage_volume = 0.1
        m_liq, U_liq = stored_state(
            "Water", pressure, 300.0, liquid_volume
        )
        m_ull, U_ull = stored_state(
            "Nitrogen", pressure, 300.0, ullage_volume
        )
        liquid_density = m_liq / liquid_volume
        ox_cda = 0.01 / (2.0 * liquid_density * 50_000.0) ** 0.5
        fuel_cda = 0.01 / (2.0 * 1000.0 * 100_000.0) ** 0.5
        network = FluidNetwork(
            nodes={
                "ox_tank": {
                    "component": "propellant_tank",
                    "geometry": TankGeometry(liquid_volume + ullage_volume),
                    "P0": pressure,
                    "state0": {
                        "P": pressure,
                        "T": 300.0,
                        "gas_T": 300.0,
                        "m_liq": m_liq,
                        "U_liq": U_liq,
                        "m_ull": m_ull,
                        "U_ull": U_ull,
                    },
                    "liquid_fluid": "Water",
                    "gas_fluid": "Nitrogen",
                },
                "ox_junction": {"model": "junction", "P0": 250_000.0},
                "fuel_source": {"model": "boundary"},
                "chamber": {
                    "component": "combustor",
                    "P0": 200_000.0,
                    "expansion_ratio": 5.0,
                    "oxidizer_fluid": "Water",
                    "fuel_fluid": "fuel",
                    "combustion_fluid": "combustion_gas",
                    "ambient_node": "ambient",
                    "cstar_efficiency": 1.0,
                    "cf_efficiency": 1.0,
                },
                "ambient": {"model": "boundary"},
            },
            branches={
                "ox_feed": {
                    "model": "incompressible_loss",
                    "fluid": "Water",
                    "from": "ox_tank",
                    "to": "ox_junction",
                    "CdA": ox_cda,
                },
                "ox_injector": {
                    "model": "incompressible_loss",
                    "fluid": "Water",
                    "from": "ox_junction",
                    "to": "chamber",
                    "CdA": ox_cda,
                },
                "fuel_injector": {
                    "model": "incompressible_loss",
                    "fluid": "fuel",
                    "from": "fuel_source",
                    "to": "chamber",
                    "CdA": fuel_cda,
                },
                "nozzle": {
                    "component": "nozzle",
                    "fluid": "combustion_gas",
                    "from": "chamber",
                    "to": "ambient",
                    "At": 2.0e-4,
                    "Cd": 1.0,
                },
            },
            fluid_properties=CoolPropPropertySource(),
            combustion_properties=CEAPropertySource(FakeCEA()),
        )
        boundaries = {
            "fuel_source": {
                "P": pressure,
                "fluids": {
                    "fuel": {
                        "phase": "liquid",
                        "rho": 1000.0,
                        "h": 100_000.0,
                    }
                },
            },
            "ambient": {"P": 100_000.0},
        }

        network.update(bcs=boundaries, commit=True)
        tank = network.nodes["ox_tank"]
        tank.state["m_liq"] = tank.dry_mass
        network.state.node["ox_tank"]["m_liq"] = tank.dry_mass

        self.assertTrue(network._apply_transitions())
        self.assertEqual(tank.mode, "gas")
        self.assertEqual(set(tank.state), {"P", "T", "m", "U"})
        self.assertAlmostEqual(tank.state["m"], m_ull)
        network.update(bcs=boundaries, commit=True)
        self.assertTrue(network._apply_transitions())
        self.assertIsInstance(
            network.nodes["chamber"].model, TwinPathJunctionModel
        )
        self.assertIsInstance(
            network.branches["nozzle"].model, TwinPathNozzleModel
        )
        self.assertTrue(all(branch.enabled for branch in network.branches.values()))

    def test_loss_component_propagates_tank_fluid_and_switches_phase_model(self):
        pressure = 300_000.0
        liquid_volume = 0.01
        ullage_volume = 0.1
        m_liq, U_liq = stored_state("Water", pressure, 300.0, liquid_volume)
        m_ull, U_ull = stored_state("Nitrogen", pressure, 300.0, ullage_volume)
        network = FluidNetwork(
            nodes={
                "tank": {
                    "component": "propellant_tank",
                    "geometry": TankGeometry(liquid_volume + ullage_volume),
                    "P0": pressure,
                    "state0": {
                        "P": pressure,
                        "T": 300.0,
                        "gas_T": 300.0,
                        "m_liq": m_liq,
                        "U_liq": U_liq,
                        "m_ull": m_ull,
                        "U_ull": U_ull,
                    },
                    "liquid_fluid": "Water",
                    "gas_fluid": "Nitrogen",
                },
                "ambient": {"model": "boundary"},
            },
            branches={
                "outlet": {
                    "component": "loss",
                    "fluid": "Water",
                    "from": "tank",
                    "to": "ambient",
                    "CdA": 1.0e-6,
                }
            },
            fluid_properties=CoolPropPropertySource(),
        )
        boundaries = {"ambient": {"P": 100_000.0}}

        liquid_result = network.update(bcs=boundaries, commit=True)
        outlet = network.branches["outlet"]
        self.assertIsInstance(outlet.model, IncompressibleLossModel)
        liquid_flow = liquid_result["branch"]["outlet"]["flows"]["main"]
        self.assertEqual(liquid_flow["fluid_name"], "Water")
        self.assertEqual(liquid_flow["fluid"]["phase"], "liquid")

        tank = network.nodes["tank"]
        tank.state["m_liq"] = tank.dry_mass
        network.state.node["tank"]["m_liq"] = tank.dry_mass
        self.assertTrue(network._apply_transitions())
        gas_result = network.update(bcs=boundaries, commit=False)

        self.assertIsInstance(outlet.model, CompressibleLossModel)
        gas_flow = gas_result["branch"]["outlet"]["flows"]["main"]
        self.assertEqual(gas_flow["fluid_name"], "Nitrogen")
        self.assertEqual(gas_flow["fluid"]["phase"], "gas")
        self.assertTrue(outlet.enabled)

    def test_fixed_head_pump_is_an_algebraic_branch(self):
        network = FluidNetwork(
            nodes={
                "source": {"model": "boundary"},
                "pump_out": {
                    "model": "junction",
                    "P0": 300_000.0,
                },
                "sink": {"model": "boundary"},
            },
            branches={
                "pump": {
                    "component": "pump",
                    "fluid": "water",
                    "from": "source",
                    "to": "pump_out",
                    "dP": 200_000.0,
                    "CdA": None,
                },
                "loss": {
                    "model": "incompressible_loss",
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
                    "fluids": {
                        "water": {
                            "phase": "liquid",
                            "rho": 1000.0,
                            "h": 100_000.0,
                        }
                    },
                },
                "sink": {"P": 100_000.0},
            },
        )

        self.assertAlmostEqual(result["node"]["pump_out"]["P"], 300_000.0)
        self.assertAlmostEqual(result["mdot"]["pump"], 1.0)

    def test_shutdown_chamber_solves_separate_gas_and_liquid_paths(self):
        properties = CoolPropPropertySource()
        gas = properties.state_pt("Nitrogen", 400_000.0, 300.0).as_dict()
        liquid = properties.state_pt("Water", 400_000.0, 300.0).as_dict()
        gas["phase"] = "gas"
        liquid["phase"] = "liquid"
        network = FluidNetwork(
            nodes={
                "gas_source": {"model": "boundary"},
                "liquid_source": {"model": "boundary"},
                "chamber": {
                    "component": "combustor",
                    "P0": 250_000.0,
                    "oxidizer_fluid": "oxidizer",
                    "fuel_fluid": "Water",
                    "combustion_fluid": "products",
                    "ambient_node": "ambient",
                    "expansion_ratio": 5.0,
                    "cstar_efficiency": 1.0,
                    "cf_efficiency": 1.0,
                },
                "ambient": {"model": "boundary"},
            },
            branches={
                "gas_feed": {
                    "component": "loss",
                    "fluid": "Nitrogen",
                    "from": "gas_source",
                    "to": "chamber",
                    "CdA": 2.0e-5,
                },
                "liquid_feed": {
                    "component": "loss",
                    "fluid": "Water",
                    "from": "liquid_source",
                    "to": "chamber",
                    "CdA": 2.0e-6,
                },
                "nozzle": {
                    "component": "nozzle",
                    "fluid": "products",
                    "from": "chamber",
                    "to": "ambient",
                    "At": 2.0e-5,
                    "Cd": 1.0,
                },
            },
            fluid_properties=properties,
        )
        chamber = network.nodes["chamber"]
        inflows = [
            branch_flow(
                "gas_feed",
                1.0,
                {
                    "components": {"Nitrogen": {**gas, "mdot": 1.0}},
                    "enabled": True,
                },
            ),
            branch_flow(
                "liquid_feed",
                1.0,
                {
                    "components": {"Water": {**liquid, "mdot": 1.0}},
                    "enabled": True,
                },
            ),
        ]
        self.assertTrue(chamber.update_mode(inflows))
        network.branches["nozzle"].set_mode(False, chamber.model.phases)

        result = network.update(
            bcs={
                "gas_source": {"P": 400_000.0, "fluids": {"Nitrogen": gas}},
                "liquid_source": {"P": 400_000.0, "fluids": {"Water": liquid}},
                "ambient": {"P": 100_000.0},
            },
            commit=False,
        )

        nozzle = result["branch"]["nozzle"]
        self.assertIsInstance(chamber.model, TwinPathJunctionModel)
        self.assertIsInstance(
            network.branches["nozzle"].model, TwinPathNozzleModel
        )
        self.assertGreater(
            sum(flow["mdot"] for flow in nozzle["flows"].values() if flow["fluid"]["phase"] == "gas"),
            0.0,
        )
        self.assertGreater(
            sum(flow["mdot"] for flow in nozzle["flows"].values() if flow["fluid"]["phase"] == "liquid"),
            0.0,
        )
        gas_mdot = sum(
            flow["mdot"] for flow in nozzle["flows"].values()
            if flow["fluid"]["phase"] == "gas"
        )
        liquid_mdot = sum(
            flow["mdot"] for flow in nozzle["flows"].values()
            if flow["fluid"]["phase"] == "liquid"
        )
        self.assertAlmostEqual(result["mdot"]["gas_feed"], gas_mdot, places=7)
        self.assertAlmostEqual(result["mdot"]["liquid_feed"], liquid_mdot, places=7)
        self.assertGreater(nozzle["gas_area_fraction"], 0.0)
        self.assertLess(nozzle["gas_area_fraction"], 1.0)
        self.assertGreater(nozzle["thrust"], 0.0)


if __name__ == "__main__":
    unittest.main()
