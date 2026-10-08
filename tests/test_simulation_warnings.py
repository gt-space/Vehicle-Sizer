import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import simulation
from diagnostics.constraints import EvaluationFailure
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


@pytest.mark.parametrize("fail", [False, True])
def test_simulation_reports_cautions_on_success_and_failure(monkeypatch, caplog, fail):
    result = SimResult()
    closed = []

    def run(*args, context, **kwargs):
        context.update(phase="runtime", flight=SimpleNamespace(result=result),
                       propulsion=SimpleNamespace(close=lambda: closed.append(True)))
        caution("test", "A nonfatal caution")
        assert not caplog.records
        if fail:
            raise RuntimeError("solver failed")
        return result

    monkeypatch.setattr(simulation, "_simulate", run)
    if fail:
        with pytest.raises(EvaluationFailure) as exc:
            simulation.simulate({})
        assert exc.value.partial_result is result
    else:
        assert simulation.simulate({}) is result
    assert closed == [True]
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
