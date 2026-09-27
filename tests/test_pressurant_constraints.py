"""Topology-independent sizing and optimizer constraint regression checks."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from Fluids.PropSystem import DesignInfeasible, PropSystem
import propulsion_fixtures as support
from propulsion_fixtures import FakeCEA, GasGeometry, GeometrySource
from test_simulation_api import make_flight
from test_flight import FakePropSystem


def build(cfg, tanks):
    with patch("Fluids.PropSystem._make_cea", return_value=FakeCEA()):
        return PropSystem(cfg, tanks)


@pytest.fixture(scope="module")
def example():
    cfg = support.config()
    tanks = dict(zip(("ox_tank", "fuel_tank", "press_tank"), support.tanks()))
    return cfg, tanks, build(cfg, tanks)


def explicit_template(example, split=False):
    cfg, tanks, system = example
    cfg, tanks = deepcopy(cfg), deepcopy(tanks)
    template = dict(circuits=deepcopy(system.circuits),
                    nodes=deepcopy(system.node_definitions),
                    branches=deepcopy(system.branch_definitions))
    # Keep only design inputs: a template must be rebound on every build.
    for node in template["nodes"].values():
        if "tank_id" in node:
            for key in ("state0", "geometry"):
                node.pop(key, None)
    for branch in template["branches"].values():
        for key in ("CdA", "design_mdot", "design_pressure", "design_temperature", "eol_pressure", "target_pressure"):
            branch.pop(key, None)
    if split:
        tanks["fuel_supply"] = GeometrySource(GasGeometry(.025))
        cfg["prop_system"]["initial_conditions"]["fuel_supply"] = dict(fluid="Nitrogen", T=290)
        cfg["tanks"]["fuel_supply"] = dict(type="pressurant", design_pressure=20e6)
        template["nodes"]["fuel_supply"] = dict(component="pressurant_tank", tank_id="fuel_supply")
        template["circuits"]["second_gas"] = dict(prop="pressurant", tank_id="fuel_supply")
        template["branches"]["FUEL_BANGBANG"].update(circuit="second_gas", **{"from": "fuel_supply"})
    # Rename every public ID; no source/destination spelling may be assumed.
    tank_names = {key: "storage_" + key for key in tanks}
    node_names = {key: "node_" + key for key in template["nodes"]}
    circuit_names = {key: "circuit_" + key for key in template["circuits"]}
    for node in template["nodes"].values():
        if "tank_id" in node:
            node["tank_id"] = tank_names[node["tank_id"]]
            if node["component"] == "pressurant_tank":
                node["P0"] = {"config": "tanks." + node["tank_id"] + ".design_pressure"}
        if "ambient_node" in node:
            node["ambient_node"] = node_names[node["ambient_node"]]
    for circuit in template["circuits"].values():
        if "tank_id" in circuit:
            circuit["tank_id"] = tank_names[circuit["tank_id"]]
    for branch in template["branches"].values():
        branch["from"], branch["to"] = node_names[branch["from"]], node_names[branch["to"]]
        branch["circuit"] = circuit_names[branch["circuit"]]
    template["nodes"] = {node_names[k]: v for k, v in template["nodes"].items()}
    template["circuits"] = {circuit_names[k]: v for k, v in template["circuits"].items()}
    template["branches"] = {"edge_" + k: v for k, v in template["branches"].items()}
    cfg["prop_system"]["initial_conditions"] = {tank_names[k]: v for k, v in cfg["prop_system"]["initial_conditions"].items()}
    cfg["tanks"] = {tank_names[k]: v for k, v in cfg["tanks"].items()}
    cfg["prop_system"]["template"] = template
    return cfg, {tank_names[k]: v for k, v in tanks.items()}


@pytest.mark.parametrize("split", [False, True])
def test_renamed_shared_and_independent_supplies(example, split):
    cfg, tanks = explicit_template(example, split)
    original = deepcopy(cfg)
    system = build(cfg, tanks)
    assert cfg == original
    assert system.chamber_id == "node_thrust_chamber"
    assert system.nozzle_id == "edge_NOZZLE"
    assert len(system.choked_branches) == 2
    assert all(value >= 0 for value in system.initial_constraints.values())
    out = system.network.update(bcs={system.ambient_id: {"P": 1e5}})
    assert len(out["constraints"]) == (6 if split else 4)
    if split:
        ox = system.branch_definitions["edge_OX_BANGBANG"]
        fuel = system.branch_definitions["edge_FUEL_BANGBANG"]
        assert ox["design_temperature"] == 260
        assert fuel["design_temperature"] == 255
        assert ox["design_pressure"] > fuel["design_pressure"]
        mass = system.initial_states["storage_fuel_supply"]["m"]
        tanks["storage_fuel_supply"] = GeometrySource(GasGeometry(.050))
        assert build(cfg, tanks).initial_states["storage_fuel_supply"]["m"] == pytest.approx(2*mass)


def test_initial_pressure_and_temperature_failures_are_constraints(example):
    cfg, tanks, _ = example
    for field, value, suffix in (("P", 3e6, "Pmin"), ("T", 210, "Tmin")):
        candidate = deepcopy(cfg)
        if field == "P":
            candidate["tanks"]["press_tank"]["design_pressure"] = value
        else:
            candidate["prop_system"]["initial_conditions"]["press_tank"][field] = value
        with pytest.raises(DesignInfeasible) as caught:
            build(candidate, tanks)
        assert caught.value.constraints[f"tank.press_tank.{suffix}"] < 0


def test_two_supplies_to_one_tank_allocate_sizing_demand(example):
    cfg, tanks = explicit_template(example, split=True)
    branches = cfg["prop_system"]["template"]["branches"]
    original = build(cfg, tanks).branch_definitions["edge_OX_BANGBANG"]["CdA"]
    branches["edge_OX_BANGBANG"]["demand_fraction"] = .4
    branches["additional_supply"] = {
        **branches["edge_OX_BANGBANG"], "from": "node_fuel_supply",
        "circuit": "circuit_second_gas", "demand_fraction": .6}
    system = build(cfg, tanks)
    assert system.branch_definitions["edge_OX_BANGBANG"]["CdA"] == pytest.approx(.4*original)
    assert len(system.choked_branches) == 3
    branches["additional_supply"]["demand_fraction"] = .5
    with pytest.raises(ValueError, match="summing to one"):
        build(cfg, tanks)


def test_design_defaults_follow_template_roles_not_tank_names(example):
    cfg, tanks = explicit_template(example, split=True)
    circuits = cfg["prop_system"]["template"]["circuits"]
    circuits["circuit_oxidizer"]["tank_id"] = "storage_ox_tank"
    circuits["circuit_fuel"]["tank_id"] = "storage_fuel_tank"
    cfg["tanks"] = {
        "storage_press_tank": {"design_pressure": 30e6, "min_temperature": 220},
        "storage_fuel_supply": {"design_pressure": 20e6, "min_temperature": 225}}
    for state in cfg["prop_system"]["initial_conditions"].values():
        state.pop("P", None)
    system = build(cfg, tanks)
    assert system.initial_states["storage_ox_tank"]["P"] == 2.6e6
    assert system.initial_states["storage_fuel_supply"]["P"] == 20e6
    mass = system.initial_states["storage_fuel_supply"]["m"]
    cfg["tanks"]["storage_fuel_supply"]["design_pressure"] = 25e6
    assert build(cfg, tanks).initial_states["storage_fuel_supply"]["m"] > mass
    assert build(cfg, tanks).initial_states["storage_fuel_supply"]["P"] == 25e6


def test_api_returns_initial_design_rejection(example):
    import simulation
    cfg, tanks, _ = example
    cfg = deepcopy(cfg)
    cfg["launch"] = {"altitude": 0, "velocity": 0}
    cfg["tanks"]["press_tank"]["design_pressure"] = 3e6
    with patch.object(simulation, "Vehicle", return_value=SimpleNamespace(tanks=tanks)), \
         patch("Fluids.PropSystem._make_cea", return_value=FakeCEA()):
        result = simulation.simulate(cfg, pure_properties=example[2].fluid_properties,
                                     combustion_properties=example[2].combustion_properties)
    assert result.termination == "infeasible_initial_design"
    assert result.feasible is False
    assert result.history is None


def test_blowdown_has_no_implicit_copv_requirements(example):
    cfg, tanks, _ = example
    cfg, tanks = deepcopy(cfg), dict(tanks)
    cfg["prop_system"]["template"] = "Configs/templates/pressure_fed_blowdown.yaml"
    del cfg["prop_system"]["initial_conditions"]["press_tank"]
    del tanks["press_tank"]
    system = build(cfg, tanks)
    assert system.initial_constraints == {}
    assert system.network.update(bcs={system.ambient_id: {"P": 1e5}})["constraints"] == {}


def test_runtime_uses_current_gamma_pressure_and_temperature_even_when_closed(example):
    system = example[2]
    nodes = {key: {"P": definition.get("P0", 1e5), "fluids": {}}
             for key, definition in system.node_definitions.items()}
    nodes["press_tank"].update(P=4e6, T=210, fluids={"Nitrogen": {"T": 210, "gamma": 1.4}})
    nodes["ox_ullage"]["P"] = 2.6e6
    nodes["fuel_ullage"]["P"] = 2.7e6
    margins = system._constraint_margins(SimpleNamespace(nodes=nodes, branches={}))
    assert margins["tank.press_tank.Tmin"] == -10
    assert margins["tank.press_tank.Pmin"] == pytest.approx(4e6 - 2.7e6/(2/2.4)**3.5)
    assert margins["branch.OX_BANGBANG.choked"] < 0


def test_flight_retains_a_violation_after_recovery():
    class Propulsion(FakePropSystem):
        def update(self, *args, **kwargs):
            out = super().update(*args, **kwargs)
            out.constraints = {"tank.supply.Tmin": -1 if len(self.calls) == 1 else 10}
            return out
    flight = make_flight(end=.3, propulsion=Propulsion())
    flight.run(record_history=False)
    assert flight.result.constraints["tank.supply.Tmin"] == -1
    assert flight.result.feasible is False
