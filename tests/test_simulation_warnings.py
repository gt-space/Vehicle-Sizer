import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import simulation
from diagnostics.errors import EvaluationFailure
from diagnostics.errors import LookupBoundsError
from FluidTables.PropertyModels import TableCombustionPropertySource
from simulation_types import SimResult
from diagnostics.warnings import caution, collect_warnings


@pytest.fixture(scope="module")
def combustion():
    return TableCombustionPropertySource(
        Path(__file__).resolve().parents[1] / "FluidTables/sizer_lookups.h5", 1)


def evaluate(source, ambient, **overrides):
    coordinates = dict(chamber_pressure=2e6, mixture_ratio=2.,
                       ambient_pressure=ambient, expansion_ratio=3.)
    coordinates.update(overrides)
    return source.evaluate(**coordinates)


def test_clamp_matches_boundary_and_reports_only_at_end(combustion, caplog):
    caplog.set_level(logging.WARNING)
    lower = float(combustion.table.axes["ambient_pressure"][0])
    reference = evaluate(combustion, lower)
    with collect_warnings() as records:
        assert evaluate(combustion, 0.) == reference
        assert evaluate(combustion, lower / 2) == reference
        assert not caplog.records
    assert len(caplog.records) == 1
    assert records[0]["code"] == "ambient_pressure_clamped"
    assert records[0]["count"] == 2
    with collect_warnings() as clean:
        evaluate(combustion, lower)
        evaluate(combustion, 101325.)
    assert clean == []


@pytest.mark.parametrize("ambient, overrides", [
    (-1., {}), (200000., {}),
    (0., {"chamber_pressure": 1.}), (0., {"mixture_ratio": .01}),
    (0., {"expansion_ratio": 10000.}),
])
def test_other_bounds_remain_strict(combustion, ambient, overrides):
    with collect_warnings(), pytest.raises(LookupBoundsError):
        evaluate(combustion, ambient, **overrides)


@pytest.mark.parametrize("ambient", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_pressure_is_not_clamped(combustion, ambient):
    with collect_warnings() as records, pytest.raises(ValueError, match="finite"):
        evaluate(combustion, ambient)
    assert records == []


@pytest.mark.parametrize("phase", [None, "runtime", "constraint assessment", "incomplete"])
def test_simulation_finalizes_metadata_and_cautions_for_every_result(monkeypatch, caplog, phase):
    result = SimResult(apogee=100, termination="apogee")
    closed = []
    pump, gg = {"pump": {"power": 10.}}, {"mass_flow": .1}
    tracking = {"tank": {"duration_s": 2.}}

    def run(*args, context, **kwargs):
        context.update(phase=phase or "runtime", flight=SimpleNamespace(result=result),
                       battery_sizing={"used_mass_kg": 1.}, design_summary={"length": 3.},
                       propulsion=SimpleNamespace(close=lambda: closed.append(True),
                           pump_sizing=pump, gg_sizing=gg,
                           network=SimpleNamespace(retry_diagnostics=[dict(rtol=1e-6)],
                                                   pressure_tracking=SimpleNamespace(records=tracking))))
        caution("test", "A nonfatal caution")
        assert not caplog.records
        if phase == "incomplete":
            result.apogee = None
            result.termination = "time_limit"
        elif phase:
            raise RuntimeError("solver failed")
        return result

    monkeypatch.setattr(simulation, "_simulate", run)
    if phase:
        with pytest.raises(EvaluationFailure) as exc:
            simulation.simulate({})
        assert exc.value.partial_result is result
        assert exc.value.phase == ("runtime" if phase == "incomplete" else phase)
    else:
        assert simulation.simulate({}) is result
    assert closed == [True]
    assert result.battery_sizing == {"used_mass_kg": 1.}
    assert result.design_summary == {"length": 3.}
    pump["pump"]["power"] = gg["mass_flow"] = tracking["tank"]["duration_s"] = -1
    assert result.pump_sizing == {"pump": {"power": 10.}}
    assert result.gg_sizing == {"mass_flow": .1}
    assert result.pressure_tracking == {"tank": {"duration_s": 2.}}
    assert result.fluid_retries == [dict(rtol=1e-6)]
    assert result.warnings == [dict(code="test", message="A nonfatal caution", count=1)]
    assert len(caplog.records) == 1


def test_nested_scopes_are_isolated():
    with collect_warnings() as outer:
        caution("outer", "outer")
        with collect_warnings() as inner:
            caution("inner", "inner")
        caution("outer", "outer")
    assert outer[0]["count"] == 2
    assert inner == [dict(code="inner", message="inner", count=1)]


def test_finalization_failure_still_closes_propulsion(monkeypatch):
    class UncopyableSizing:
        def __deepcopy__(self, memo):
            raise RuntimeError('metadata failure')

    closed = []
    def run(*args, context, **kwargs):
        context['propulsion'] = SimpleNamespace(pump_sizing=UncopyableSizing(),
                                               close=lambda: closed.append(True))
        return SimResult(apogee=100., termination='apogee')

    monkeypatch.setattr(simulation, '_simulate', run)
    with pytest.raises(RuntimeError, match='metadata failure'):
        simulation.simulate({})
    assert closed == [True]
