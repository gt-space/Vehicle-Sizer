from types import SimpleNamespace

import pytest

from constraints import DesignInfeasible, finalize, vehicle_limit_margins
from simulation_types import SimResult
from optimizer import candidate_score


@pytest.mark.parametrize('length,margin', [(5., 5.), (6.25, 0.)])
def test_ld_pass_and_boundary(length, margin):
    cfg = dict(constraints={'max_length_to_diameter': 25.}, vehicle={'OMLD': .25})
    margins = vehicle_limit_margins(cfg, SimpleNamespace(length=length))
    assert margins['max_length_to_diameter'] == margin
    result = finalize(SimResult(constraints=margins), cfg['constraints'])
    assert result.constraint_records['max_length_to_diameter'].units == '1'
    assert result.constraint_records['max_length_to_diameter'].phase == 'sizing'


def test_ld_reject_preserves_sizing_margin_and_score():
    cfg = dict(constraints={'max_length_to_diameter': 25.}, vehicle={'OMLD': .25})
    with pytest.raises(DesignInfeasible) as caught:
        vehicle_limit_margins(cfg, SimpleNamespace(length=7.5))
    result = finalize(SimResult(termination='infeasible_initial_design',
        constraints=caught.value.constraints), cfg['constraints'])
    assert result.constraints['max_length_to_diameter'] == -5.
    assert not result.accepted
    score, parts = candidate_score(result, dict(constraint_scales={'max_length_to_diameter': 5.}, unit_scales={}))
    assert score == 2.5
    assert parts == {'max_length_to_diameter': 1.}


def test_ld_optional_and_missing_not_passing():
    assert vehicle_limit_margins({}, object()) == {}
    result = finalize(SimResult(), {'max_length_to_diameter': 25.})
    assert result.constraints['max_length_to_diameter'] is None
    assert not result.accepted
