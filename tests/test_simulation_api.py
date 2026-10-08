from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from Fluids.design import initial_conditions, tank_design_pressure
from Flight.Flight import FlightSim
from simulation_types import KinematicsState, PlantOut, SimResult
from Vehicle.Engine import Engine
from Vehicle.Material import MaterialProperties
from Vehicle.Vehicle import Vehicle
from Vehicle.geometry_constraints import (check_press_tank_to_airframe, check_feedline,
                                 check_nozzle, check_engine_length)
from Vehicle.sections.PropTank import PropTank
from test_flight import FakeAero, FakeEnvironment, FakePropSystem, FakeVehicle


def make_flight(end=0.1, propulsion=None):
    return FlightSim(
        {"simulation": {"dt": 0.1, "t_end": end},
         "launch": {"altitude": 0.0, "rail_height": 0.0}},
        FakeEnvironment(), FakeAero(), propulsion or FakePropSystem(), FakeVehicle())


def test_headless_skips_loads_and_vector_snapshots_and_matches_history():
    flight = make_flight()
    with patch.object(flight.loads, "evaluate", side_effect=AssertionError), \
         patch.object(flight, "mass_properties", side_effect=AssertionError):
        assert flight.run(record_history=False, compute_loads=False) == []
    reference = make_flight()
    reference.run()
    assert flight.result.max_altitude == reference.result.max_altitude
    assert flight.result.history is None
    assert flight.result.apogee is None
    assert flight.result.termination == "time_limit"
    assert not flight.result.burn_complete


def test_coast_reports_interpolated_apogee_not_time_limit_maximum():
    class Coast(FakePropSystem):
        def update(self, *args, **kwargs):
            out = super().update(*args, **kwargs)
            out.propulsion = replace(out.propulsion, thrust=0.0, mode="shutdown")
            return out
    flight = make_flight(end=4, propulsion=Coast())
    flight.run(h0=100, v0=10, record_history=False)
    result = flight.result
    assert result.apogee_reached
    assert result.termination == "apogee"
    assert result.apogee == pytest.approx(105.1, abs=0.03)
    assert result.max_altitude == result.apogee
    assert result.apogee_time <= result.final_time


def test_pitch_projects_thrust_not_body_load():
    flight = make_flight()
    atmosphere = FakeEnvironment.atmosphere(10, 10)
    fluid = FakePropSystem().update(None, atmosphere, None)
    kin = KinematicsState(t=0, dt=1, x=0., h=10, vx=0., vz=10, theta=np.pi/6, q=0, alpha=0.0, m=16, Iyy=2)
    plant = PlantOut(FakeAero.evaluate(kin, atmosphere, True), None, fluid)
    force = flight.forces(kin, plant, kin.m)
    assert force["thrust"] == 300
    assert force["Fz"] + force["gravity"] == pytest.approx(150)
    assert force["net"] == pytest.approx(150 - force["gravity"])


def test_initial_defaults_are_design_based_without_mutation():
    cfg = {"prop_system": {
        "template": {"circuits": {}, "branches": {}, "nodes": {
            "chamber": {"component": "boundary", "P0": {"config": "prop_system.Pc_target"}},
            "ox": {"component": "propellant_tank", "tank_id": "ox_tank", "P0": {"node": "chamber", "relative_rise": .2, "rise": 1e5}},
            "fuel": {"component": "propellant_tank", "tank_id": "fuel_tank", "P0": {"node": "chamber", "relative_rise": .1, "rise": 2e5}},
            "gas": {"component": "pressurant_tank", "tank_id": "press_tank", "P0": {"config": "tanks.press_tank.design_pressure"}}}},
        "Pc_target": 2e6,
        "ox_inj_stiffness": .2, "fuel_inj_stiffness": .1,
        "ox_tank_inj_dp": 1e5, "fuel_tank_inj_dp": 2e5,
        "initial_conditions": {
            "ox_tank": {"fluid": "Oxygen", "gas_fluid": "Nitrogen", "T": 95, "gas_T": 290},
            "fuel_tank": {"fluid": "n-Dodecane", "T": 295},
            "press_tank": {"fluid": "Nitrogen", "T": 300}}},
        "tanks": {"press_tank": {"type": "pressurant", "design_pressure": 30e6}}}
    original = deepcopy(cfg)
    first = initial_conditions(cfg)
    assert cfg == original
    assert first["ox_tank"]["P"] == tank_design_pressure(cfg, "ox_tank")
    assert first["fuel_tank"]["T"] == 295
    assert first["ox_tank"]["gas_T"] == 290
    assert first["press_tank"]["P"] == 30e6
    cfg["prop_system"]["Pc_target"] *= 2
    assert initial_conditions(cfg)["ox_tank"]["P"] > first["ox_tank"]["P"]
    for tank_id in ("ox_tank", "fuel_tank"):
        cfg["prop_system"]["initial_conditions"][tank_id]["P"] = 3e6
    resolved = initial_conditions(cfg)
    assert resolved["ox_tank"]["P"] == resolved["fuel_tank"]["P"] == 3e6
    assert resolved["press_tank"]["P"] == 30e6


@pytest.mark.parametrize("name", ["press_tank", "arbitrary_supply"])
@pytest.mark.parametrize("pressure", [30e6, 20e6])
def test_pressurant_pressure_is_one_design_and_initial_input(name, pressure):
    cfg = {"tanks": {name: {"type": "pressurant", "design_pressure": pressure}},
           "prop_system": {"initial_conditions": {name: {"fluid": "Nitrogen", "T": 300}}}}
    assert initial_conditions(cfg)[name]["P"] == pressure
    assert "P" not in cfg["prop_system"]["initial_conditions"][name]


@pytest.mark.parametrize("field", ["T", "gas_T"])
@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf"), "missing"])
def test_initial_temperatures_are_explicit_and_valid(field, value):
    state = {"fluid": "Oxygen", "gas_fluid": "Nitrogen", "P": 2e6, "T": 95, "gas_T": 290}
    if value == "missing":
        del state[field]
    else:
        state[field] = value
    with pytest.raises(ValueError, match=field):
        initial_conditions({"prop_system": {"initial_conditions": {"ox_tank": state}}})


def test_copv_default_does_not_require_an_engine_pressure_ladder():
    cfg = {"prop_system": {"initial_conditions": {"press_tank": {"fluid": "Nitrogen", "T": 300}}},
           "tanks": {"press_tank": {"type": "pressurant", "design_pressure": 30e6}}}
    assert initial_conditions(cfg)["press_tank"]["P"] == 30e6
    del cfg["tanks"]["press_tank"]["design_pressure"]
    with pytest.raises(ValueError, match="design_pressure"):
        initial_conditions(cfg)


@pytest.mark.parametrize("field", ["initial_temperature", "sink_temperature"])
def test_thermal_temperatures_are_explicit(field):
    from Thermals import ThermalNetwork
    temperatures = {"initial_temperature": 300, "sink_temperature": 290}
    del temperatures[field]
    with pytest.raises(ValueError, match=field):
        ThermalNetwork({"thermal": temperatures}, SimpleNamespace(sections=[]))


def test_weld_efficiency_and_geometry_share_exact_cell_extent():
    def tank(efficiency):
        return PropTank({"vehicle": {"dx": .037, "OMLD": .3}},
                        60, 1000, MaterialProperties("test", 2700, 276e6, 77e9),
                        None, 2.7e6, .001, .04, .002, 1.75, 1.1, "tank",
                        weld_efficiency=efficiency)
    base, welded = tank(1), tank(.8)
    assert welded.wall_thickness > base.wall_thickness
    assert welded.local_edges[-1] == welded.length
    assert welded.n * welded.dx == pytest.approx(welded.length)
    np.testing.assert_allclose(welded.local_edges, welded.get_fluid_geometry()._profile["height"])
    with pytest.raises(ValueError, match="weld_efficiency"):
        tank(0)


def test_engine_overlay_is_conservative_and_survives_mass_assembly():
    engine = Engine(30, .55, .02)
    edges = np.array([0, .3, .6, .9, 1.2])
    overlay = engine.mass_on_grid(edges, .6)
    np.testing.assert_allclose(overlay, [0, 0, 30*.3/.55, 30*.25/.55])
    vehicle = Vehicle.__new__(Vehicle)
    vehicle.engine_mass = overlay
    section = SimpleNamespace(station=(edges[:-1]+edges[1:])/2, mass=np.ones(4),
                              EI=np.ones(4), lat_area=np.ones(4), surf_area=np.ones(4))
    vehicle.sections = [section]
    vehicle._assemble_vectors()
    assert vehicle.mass.sum() == pytest.approx(34)
    section.mass = np.full(4, 2)
    vehicle._assemble_vectors()
    assert vehicle.mass.sum() == pytest.approx(38)
    with pytest.raises(ValueError, match="complete engine"):
        engine.mass_on_grid(edges, 1)


def test_geometry_constraints_have_consistent_sign():
    tank_section = SimpleNamespace(diameter=.2)
    assert check_press_tank_to_airframe(tank_section, .3) > 0
    assert check_press_tank_to_airframe(tank_section, .2) == 0
    assert check_press_tank_to_airframe(tank_section, .19) < 0
    tank = SimpleNamespace(passthrough_diameter=.1, passthrough_wall_thickness=.01)
    assert check_feedline(tank, .07) > 0
    assert check_feedline(tank, .09) < 0
    fin = SimpleNamespace(length=.5, boattail_aft_diameter=.2, wall_thickness=.01)
    engine = Engine(30, .6, np.pi*.1**2)
    assert check_nozzle(fin, engine) < 0
    assert check_engine_length(fin, engine) < 0
    assert SimResult(geometry_constraints={"missing": None}).feasible is None
    assert SimResult(geometry_constraints={"failed": -1, "missing": None}).feasible is False


@pytest.mark.parametrize("apogee, expected", [(None, "not reached"), (105.0, "105.0 m")])
def test_cli_distinguishes_apogee_from_maximum(monkeypatch, capsys, apogee, expected):
    import sys
    import main

    result = SimResult(max_altitude=105, apogee=apogee,
                       history=[{"kinematics": SimpleNamespace(t=1)}])
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr(main, "load_config", lambda _: {
        "simulation": {"output": "unused.csv", "plot": "unused.png"}})
    monkeypatch.setattr(main, "simulate", lambda *args, **kwargs: result)
    monkeypatch.setattr(main, "history_rows", lambda _: [])
    monkeypatch.setattr(main, "write_history", lambda *args: None)
    monkeypatch.setattr(main, "write_events", lambda *args: None)
    monkeypatch.setattr(main, "write_structural_loads", lambda *args: None)
    monkeypatch.setitem(sys.modules, "flight_plots", SimpleNamespace(plot_flight=lambda *args: {}))
    main.main()
    output = capsys.readouterr().out
    assert "Max altitude reached: 105.0 m" in output
    assert f"Apogee: {expected}" in output
