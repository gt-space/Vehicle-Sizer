from types import SimpleNamespace

import pytest

from diagnostics.constraints import finalize, vehicle_limit_records
from simulation_types import SimResult
from optimizer.core import candidate_score


@pytest.mark.parametrize('length,margin', [(5., 5.), (6.25, 0.)])
def test_ld_pass_and_boundary(length, margin):
    cfg = dict(constraints={'max_length_to_diameter': 25.}, vehicle={'OMLD': .25})
    margins = vehicle_limit_records(cfg, SimpleNamespace(length=length))
    assert margins['max_length_to_diameter'].margin == margin
    result = finalize(SimResult(constraint_records=margins), cfg['constraints'])
    assert result.constraint_records['max_length_to_diameter'].units == '1'
    assert result.constraint_records['max_length_to_diameter'].phase == 'sizing'


def test_ld_reject_preserves_sizing_margin_and_score():
    cfg = dict(constraints={'max_length_to_diameter': 25.}, vehicle={'OMLD': .25})
    margins = vehicle_limit_records(cfg, SimpleNamespace(length=7.5))
    result = finalize(SimResult(termination='infeasible_initial_design',
        constraint_records=margins), cfg['constraints'])
    assert result.constraint_records['max_length_to_diameter'].margin == -5.
    assert not result.accepted
    score, parts, penalties = candidate_score(result, dict(constraint_scales={'max_length_to_diameter': 5.}, unit_scales={}))
    assert score == 2.5
    assert parts == {'max_length_to_diameter': 1.}


def test_ld_optional_and_missing_not_passing():
    assert vehicle_limit_records({}, object()) == {}
    result = finalize(SimResult(), {'max_length_to_diameter': 25.})
    assert result.constraint_records['max_length_to_diameter'].margin is None
    assert not result.accepted
