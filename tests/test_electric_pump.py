from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from Fluids.design import pump_definition, size_electric_pump
from Fluids.helpers.templates import load_template
from Fluids.PropSystem import PropSystem, DesignInfeasible
import propulsion_fixtures as support


def config():
    cfg = support.config()
    prop = cfg["prop_system"]
    prop.update(template="Configs/templates/pump_fed_blowdown.yaml",
                pumps={name: dict(drive="electric", pressure_rise_pa=1e6,
                                 efficiency=.7, max_power_kw=15, gas_CdA=1e-4)
                       for name in ("oxidizer_pump", "fuel_pump")})
    del prop["initial_conditions"]["press_tank"]
    del cfg["tanks"]["press_tank"]
    for leg in ("ox", "fuel"):
        prop[f"{leg}_inj_pumpout_dp"] = 2e5
        prop[f"{leg}_pumpin_tank_dp"] = 2e5
    return cfg


def build(cfg):
    ox, fuel, _ = support.tanks()
    with patch("Fluids.PropSystem._make_cea", return_value=support.FakeCEA()):
        return PropSystem(cfg, {"ox_tank": ox, "fuel_tank": fuel})


def test_power_formula_and_scaling():
    definition = dict(pressure_rise_pa=1e6, efficiency=.5, max_power_kw=15)
    sized = size_electric_pump(definition, 5, 1000)
    assert sized["volume_flow_m3_s"] == .005
    assert sized["required_power_kw"] == 10
    assert sized["power_margin_kw"] == 5
    assert size_electric_pump(definition, 10, 1000)["required_power_kw"] == 20
    assert size_electric_pump({**definition, "efficiency": 1}, 5, 1000)["required_power_kw"] == 5
    assert size_electric_pump({**definition, "pressure_rise_pa": 1.5e6}, 5, 1000)["power_margin_kw"] == 0


@pytest.mark.parametrize("key,value", [("efficiency", 0), ("efficiency", 1.1),
    ("efficiency", float("nan")), ("pressure_rise_pa", -1),
    ("max_power_kw", float("inf")), ("gas_CdA", 0), ("drive", "unknown")])
def test_invalid_inputs(key, value):
    prop = config()["prop_system"]
    prop["pumps"]["oxidizer_pump"][key] = value
    with pytest.raises(ValueError):
        pump_definition(prop, "oxidizer_pump")


def test_sizing_and_runtime_share_head_without_mutating_inputs():
    cfg = config()
    original = deepcopy(cfg)
    system = build(cfg)
    assert cfg == original
    sized = system.pump_sizing["oxidizer_pump"]
    assert sized["design_mdot"] == system.mdot_ox
    assert sized["inlet_pressure_pa"] == load_template(cfg)["nodes"]["ox_pump_in"]["P0"]
    assert sized["inlet_pressure_pa"] != cfg["prop_system"]["initial_conditions"]["ox_tank"]["P"]
    assert system.network.branches["OX_PUMP"].parameters["dP"] == sized["pressure_rise_pa"]
    assert system.initial_constraints["pump.oxidizer_pump.max_power"] == sized["power_margin_kw"]
    output = system.update(None, SimpleNamespace(p=1e5), {})
    assert output.constraints["pump.oxidizer_pump.max_power"] == sized["power_margin_kw"]
    cfg["prop_system"]["pumps"]["oxidizer_pump"]["pressure_rise_pa"] = 1.2e6
    changed = build(cfg)
    assert changed.pump_sizing["oxidizer_pump"]["required_power_kw"] > sized["required_power_kw"]
    assert changed.network.branches["OX_PUMP"].parameters["dP"] == 1.2e6


def test_explicit_template_uses_pump_references_not_branch_names():
    cfg = config()
    system = build(cfg)
    template = dict(circuits=deepcopy(system.circuits), nodes=deepcopy(system.node_definitions),
                    branches=deepcopy(system.branch_definitions))
    for node in template["nodes"].values():
        if "tank_id" in node:
            node.pop("geometry")
            node.pop("state0")
    for branch in template["branches"].values():
        if branch["component"] == "pump":
            branch.pop("dP")
            branch.pop("gas_CdA")
    template["branches"]["renamed_pump_branch"] = template["branches"].pop("OX_PUMP")
    cfg["prop_system"]["template"] = template
    changed = build(cfg)
    assert changed.pump_sizing == system.pump_sizing
    assert changed.network.branches["renamed_pump_branch"].parameters["dP"] == 1e6


def test_exact_power_limit_passes():
    cfg = config()
    baseline = build(cfg)
    for name, sizing in baseline.pump_sizing.items():
        cfg["prop_system"]["pumps"][name]["max_power_kw"] = sizing["required_power_kw"]
    assert all(value == 0 for value in build(cfg).sizing_constraints.values())


def test_all_failed_pump_limits_return_before_network_construction():
    cfg = config()
    for pump in cfg["prop_system"]["pumps"].values():
        pump["max_power_kw"] = .001
    with patch("Fluids.PropSystem.FluidNetwork.__init__", side_effect=AssertionError("Runtime network created")):
        with pytest.raises(DesignInfeasible) as caught:
            build(cfg)
    assert len(caught.value.constraints) == 2
    assert all(v < 0 for v in caught.value.constraints.values())
    assert len(caught.value.pump_sizing) == 2


@pytest.mark.parametrize("failed", [False, True])
def test_api_preserves_pump_margins_and_sizing(failed):
    import simulation
    from simulation_types import SimResult
    cfg = config()
    reference = build(cfg)
    if failed:
        for pump in cfg["prop_system"]["pumps"].values():
            pump["max_power_kw"] = .001
    cfg.update(launch=dict(altitude=0, velocity=0),
               environment=dict(max_altitude=1000, altitude_step=100), aero={})
    cfg["engine"].update(mass=30, length=.5)
    ox, fuel, _ = support.tanks()
    vehicle = SimpleNamespace(tanks={"ox_tank": ox, "fuel_tank": fuel}, battery_sizing={},
                              build=lambda engine: None, aero_candidate=lambda: {})
    flight = SimpleNamespace(run=lambda **kwargs: None, result=SimResult())
    with patch.object(simulation, "Vehicle", return_value=vehicle), \
         patch.object(simulation, "FlightSim", return_value=flight) as flight_factory, \
         patch.object(simulation, "Aero"), patch.object(simulation, "Environment"):
        result = simulation.simulate(cfg, pure_properties=reference.fluid_properties,
                                     combustion_properties=reference.combustion_properties,
                                     aero_model=object())
    assert len(result.pump_sizing) == 2
    assert result.feasible is (not failed)
    if failed:
        flight_factory.assert_not_called()
        assert result.termination == "infeasible_initial_design"
    assert result.constraints["pump.oxidizer_pump.max_power"] == result.pump_sizing["oxidizer_pump"]["power_margin_kw"]
