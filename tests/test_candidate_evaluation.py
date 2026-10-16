from diagnostics.constraints import ConstraintRecord
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np

import simulation
from diagnostics.constraints import configured_limits, finalize, merge_records
from diagnostics.errors import EvaluationFailure, GeometryError
from simulation_types import KinematicsState, SimResult
from test_flight import FakePropSystem
from test_simulation_api import make_flight


def test_acceptance_requires_completed_mission_and_only_max_burn_limit():
    limits = {"goal_apogee": 100, "max_burn_duration": 2}
    for burn in (0.5, 2):
        result = finalize(SimResult(apogee=100, burn_duration=burn, burn_complete=True), limits)
        assert result.completed and result.accepted
    assert not finalize(SimResult(apogee=99, burn_complete=True), limits).accepted
    assert not finalize(SimResult(max_altitude=200, burn_complete=True), limits).accepted
    result = finalize(SimResult(apogee=100, burn_duration=3, burn_complete=True), limits)
    assert result.constraint_records["max_burn_duration"].margin == -1
    assert not result.accepted
    assert not finalize(SimResult(apogee=100, burn_duration=1), limits).accepted
    assert not SimResult(apogee=100, burn_complete=True).accepted
    with pytest.raises(ValueError, match="min_burn_duration"):
        configured_limits({"constraints": {"min_burn_duration": 1}})


def test_optional_geometry_does_not_block_acceptance():
    result = finalize(SimResult(apogee=100, burn_complete=True,
                                constraint_records={"feedline.routing": ConstraintRecord(None, "Vehicle.geometry_constraints", "sizing", "m", required=False)}),
                      {"goal_apogee": 100, "max_burn_duration": 2})
    assert result.accepted
    assert not result.constraint_records["feedline.routing"].required


@pytest.mark.parametrize("error", [GeometryError({"engine.length": -0.2}), RuntimeError("solver failed")])
def test_all_preflight_failures_use_one_typed_error_contract(error):
    cfg = {"constraints": {"ignored_unknown_limit": -1},
           "prop_system": {"template": dict(circuits={}, nodes={}, branches={})}}
    with patch.object(simulation, "Vehicle", side_effect=error), patch.object(simulation, "FlightSim") as flight:
        with pytest.raises(EvaluationFailure) as caught:
            simulation.simulate(cfg, pure_properties=object(), combustion_properties=object())
    flight.assert_not_called()
    assert caught.value.__cause__ is error
    assert caught.value.phase == 'sizing'
    assert caught.value.partial_result is None
    assert caught.value.diagnostic['exception'] == type(error).__name__


@pytest.mark.parametrize('assess', [False, True])
def test_single_build_shared_physics_and_opt_in_assessment(assess):
    cfg = {"launch": {"altitude": 0, "velocity": 0}, "engine": {"mass": 1, "length": 1},
           "vehicle": {"OMLD": .25},
           "prop_system": {"template": dict(circuits={}, nodes={}, branches={})},
           "constraints": {"max_length_to_diameter": 1, "goal_apogee": 1000, "max_burn_duration": 1},
           "environment": {"max_altitude": 1000, "altitude_step": 100}, "aero": {}}
    vehicle = SimpleNamespace(tanks={}, sections=[], cfg=cfg, length=2., battery_sizing={}, aero_candidate=lambda: {})
    propulsion = SimpleNamespace(exit_area=1, pump_sizing={}, gg_sizing={}, sizing_constraints={},
                                 aero_nodes=frozenset(), node_definitions={}, branch_definitions={}, tank_definitions={}, close=lambda: None,
                                 network=SimpleNamespace(constraint_records={}, pressure_tracking=SimpleNamespace(records={}), retry_diagnostics=[]))
    result = SimResult(apogee=100, burn_complete=True, burn_duration=2, termination='apogee')
    flight = SimpleNamespace(result=result, run=lambda **kwargs: result.constraint_records.update(propulsion.network.constraint_records))
    with patch.object(simulation, 'Vehicle', return_value=vehicle) as build_vehicle, \
         patch.object(simulation, 'PropSystem', return_value=propulsion) as build_propulsion, \
         patch.object(vehicle, 'build', create=True), patch.object(simulation, 'Aero'), \
         patch.object(simulation, 'FlightSim', return_value=flight):
        output = simulation.simulate(cfg, pure_properties=object(), combustion_properties=object(),
                                     aero_model=object(), assess_constraints=assess)
    build_vehicle.assert_called_once()
    build_propulsion.assert_called_once()
    assert output is result and output.completed
    if assess:
        assert output.constraint_records['goal_apogee'].margin == -900
        assert output.constraint_records['max_length_to_diameter'].margin == -7
        assert output.feasible is False
    else:
        assert output.constraint_records == {} and output.constraint_records == {}
        assert output.feasible is None


@pytest.mark.parametrize('termination', ['time_limit', 'no_ascent'])
def test_incomplete_flight_is_a_failure_with_partial_state(monkeypatch, termination):
    result = SimResult(termination=termination, final_time=3)
    monkeypatch.setattr(simulation, '_simulate', lambda *a, **k: result)
    with pytest.raises(EvaluationFailure) as caught:
        simulation.simulate({})
    assert caught.value.partial_result is result
    assert caught.value.diagnostic['kind'] == 'incomplete_flight'
    assert caught.value.diagnostic['details']['termination'] == termination


def test_nonfinite_constraint_is_fatal_not_an_infeasible_margin():
    with pytest.raises(ValueError, match="Nonfinite"):
        ConstraintRecord(float('nan'), 'test', 'runtime', 'K')
    records = {}
    for time, value in enumerate((2, -1, 1)):
        merge_records(records, {'x': ConstraintRecord(value, 'test', 'runtime', 'K')}, time=time)
    assert records['x'].margin == -1 and records['x'].time == 1


def test_headless_loads_match_history_and_twr_is_tracked():
    headless, recorded = make_flight(), make_flight()
    for flight in (headless, recorded):
        flight.cfg["launch"]["rail_height"] = 15
    headless.run(record_history=False, compute_loads=True)
    recorded.run(record_history=True, compute_loads=True)
    assert headless.result.load_peaks == recorded.result.load_peaks
    assert headless.result.load_peaks
    assert headless.result.history is None
    assert headless.result.min_rail_twr == recorded.result.min_rail_twr
    assert headless.result.min_rail_twr_time == 0
    assert recorded.result.history[0]["forces"]["twr"] > 0


def test_twr_interpolates_first_exit_and_does_not_reenter():
    flight = make_flight()
    flight.cfg["launch"]["rail_height"] = 5
    result = SimResult()
    start = KinematicsState(t=0, dt=1, x=0., h=0, vx=0., vz=1, theta=np.pi/2, q=0, alpha=0, m=10, Iyy=1)
    end = replace(start, t=1, h=10)
    flight._record_twr(result, start, 2)
    flight._record_twr(result, end, 0, start, 2)
    assert result.min_rail_twr == 1
    assert result.rail_exit_time == result.min_rail_twr_time == .5
    flight._record_twr(result, replace(start, t=2), 0, end, 0)
    assert result.min_rail_twr == 1
    zero = SimResult()
    flight._record_twr(zero, start, 0)
    assert zero.min_rail_twr == 0


def test_exceeded_burn_limit_does_not_stop_flight():
    class TimedBurn(FakePropSystem):
        time = 0

        def update(self, dt, *args, **kwargs):
            out = super().update(dt, *args, **kwargs)
            self.time += dt or 0
            if self.time >= .3:
                out.propulsion = replace(out.propulsion, mode="shutdown", thrust=0,
                                         shutdown_reason="test_shutdown")
            return out

    flight = make_flight(end=2, propulsion=TimedBurn())
    limits = {"goal_apogee": 0, "max_burn_duration": .05}
    flight.cfg["constraints"] = limits
    flight.run(record_history=False, compute_loads=False)
    result = finalize(flight.result, limits)
    assert result.completed and result.burn_complete
    assert result.final_time > result.burn_duration > .05
    assert result.constraint_records["max_burn_duration"].margin < 0
    assert not result.accepted


def test_canonical_records_preserve_policy_time_and_snapshot_isolation():
    from dataclasses import FrozenInstanceError
    from diagnostics.constraints import feasibility
    required = ConstraintRecord(None, 'custom-component', 'runtime', 'Pa')
    optional = ConstraintRecord(-100., 'optional-check', 'sizing', 'm', required=False)
    records = {'optional': optional}
    merge_records(records, {'required': required}, time=10.)
    assert records['required'].time is None
    assert feasibility(records) is None
    merge_records(records, {'required': replace(required, margin=2.)}, time=1.)
    snapshot = dict(records)
    assert feasibility(records) is True
    merge_records(records, {'required': replace(required, margin=-1.)}, time=2.)
    merge_records(records, {'required': replace(required, margin=3.)}, time=3.)
    assert feasibility(records) is False
    assert records['required'] == replace(required, margin=-1., time=2.)
    assert snapshot['required'] == replace(required, margin=2., time=1.)
    with pytest.raises(FrozenInstanceError):
        records['required'].margin = 0.
    result = finalize(SimResult(apogee=100., burn_complete=True),
                      {'goal_apogee': 100., 'max_burn_duration': 2.})
    result.constraint_records.update(snapshot)
    assert result.accepted
    result.constraint_records['required'] = required
    assert not result.accepted
