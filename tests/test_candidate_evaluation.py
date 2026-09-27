from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import simulation
from constraints import (DesignInfeasible, EvaluationFailure, GeometryError, OperatingInfeasible,
                         configured_limits, finalize, merge_margins)
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
    assert result.constraints["max_burn_duration"] == -1
    assert not result.accepted
    assert not finalize(SimResult(apogee=100, burn_duration=1), limits).accepted
    assert not SimResult(apogee=100, burn_complete=True).accepted
    with pytest.raises(ValueError, match="min_burn_duration"):
        configured_limits({"constraints": {"min_burn_duration": 1}})


def test_optional_geometry_does_not_block_acceptance():
    result = finalize(SimResult(apogee=100, burn_complete=True,
                                geometry_constraints={"feedline.routing": None}),
                      {"goal_apogee": 100, "max_burn_duration": 2})
    assert result.accepted
    assert not result.constraint_records["feedline.routing"].required


@pytest.mark.parametrize("error", [DesignInfeasible({"pump.p.max_power": -2}),
                                        GeometryError({"engine.length": -0.2})])
def test_preflight_reject_returns_without_flight(error):
    cfg = {"launch": {"altitude": 0, "velocity": 0},
           "constraints": {"goal_apogee": 100, "max_burn_duration": 2}}
    with patch.object(simulation, "Vehicle", side_effect=error), \
         patch.object(simulation, "FlightSim") as flight:
        result = simulation.simulate(cfg, pure_properties=object(), combustion_properties=object())
    flight.assert_not_called()
    assert result.termination == "infeasible_initial_design"
    assert result.feasible is False
    assert not result.completed and not result.accepted
    assert result.initial_mass is None
    assert result.constraints["goal_apogee"] is None
    record = result.constraint_records[next(iter(error.constraints))]
    assert record.phase == "sizing" and record.margin < 0


def test_geometry_build_rejection_and_runtime_failure_are_distinct():
    cfg = {"launch": {"altitude": 0, "velocity": 0}, "engine": {"mass": 1, "length": 1},
           "environment": {"max_altitude": 1000, "altitude_step": 100}, "aero": {}}
    vehicle = SimpleNamespace(tanks={}, aero_candidate=lambda: {})
    propulsion = SimpleNamespace(exit_area=1, pump_sizing={}, sizing_constraints={}, close=lambda: None)
    vehicle.build = lambda engine: None
    partial = SimResult(final_time=1)
    flight = SimpleNamespace(result=partial)
    with patch.object(simulation, "Vehicle", return_value=vehicle), \
         patch.object(simulation, "PropSystem", return_value=propulsion), \
         patch.object(simulation, "Aero"), \
         patch.object(simulation, "FlightSim", return_value=flight) as factory:
        with patch.object(vehicle, "build", side_effect=GeometryError({"engine.length": -1})):
            result = simulation.simulate(cfg, pure_properties=object(), combustion_properties=object(), aero_model=object())
            assert result.geometry_constraints["engine.length"] == -1
            factory.assert_not_called()
        failure = RuntimeError("solver diverged")
        with patch.object(flight, "run", side_effect=failure, create=True):
            with pytest.raises(EvaluationFailure) as caught:
                simulation.simulate(cfg, pure_properties=object(), combustion_properties=object(), aero_model=object())
        assert caught.value.__cause__ is failure
        assert caught.value.phase == "runtime"
        assert caught.value.partial_result is partial
        assert caught.value.config == cfg and caught.value.config is not cfg
        operating = OperatingInfeasible({"node.fuel_pump_in.Pmin": -3239.}, time=2.)
        with patch.object(flight, "run", side_effect=operating, create=True):
            result = simulation.simulate(cfg, pure_properties=object(),
                                         combustion_properties=object(), aero_model=object())
        assert result is partial and result.final_time == 1
        assert result.termination == "infeasible_operating_state"
        assert not result.completed and not result.accepted
        record = result.constraint_records["node.fuel_pump_in.Pmin"]
        assert (record.margin, record.phase, record.units, record.time) == (-3239., "runtime", "Pa", 2.)


def test_nonfinite_constraint_is_fatal_not_an_infeasible_margin():
    with pytest.raises(ValueError, match="Nonfinite"):
        merge_margins({}, {"tank.x.Tmin": float("nan")})
    margins, times = {}, {}
    merge_margins(margins, {"x": 2}, times=times, time=0)
    merge_margins(margins, {"x": -1}, times=times, time=1)
    merge_margins(margins, {"x": 1}, times=times, time=2)
    assert margins == {"x": -1} and times == {"x": 1}


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
    start = KinematicsState(0, 1, 0, 1, 0, 0, 10, 1)
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
    assert result.constraints["max_burn_duration"] < 0
    assert not result.accepted
