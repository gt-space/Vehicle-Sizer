"""Unsupported volume phase changes remain unresolved without aborting a search."""
from copy import deepcopy

import pytest

from diagnostics.constraints import EvaluationFailure, finalize
from diagnostics.errors import UnsupportedPhaseChangeError, ModelDomainExceeded, failure_details
from Fluids.FluidNode import VolumeComponent, PropellantTankComponent
from optimizer.core import FailureMonitor, candidate_class, candidate_score
from simulation_types import SimResult


@pytest.mark.parametrize('component', [VolumeComponent, PropellantTankComponent])
@pytest.mark.parametrize('event,source,target', [
    ('condense', 'gas', 'saturated'),
    ('evaporate', 'saturated', 'gas'),
    ('liquid_limit', 'saturated', 'liquid'),
])
def test_all_volume_thermodynamic_transitions_reject_before_state_changes(component, event, source, target):
    node = component.__new__(component)
    node.id, node.mode = 'arbitrary_volume', source
    node.definition = dict(fluid='Nitrogen', gas_fluid='Nitrogen')
    values = dict(m=1., U=200000., P=2e5, quality=1.)
    original = deepcopy(values)
    with pytest.raises(UnsupportedPhaseChangeError) as failure:
        node.apply_event(event, values, 12.5)
    assert node.mode == source
    assert values == original
    error = failure.value
    assert error.details == dict(node_id=node.id, fluid='Nitrogen', from_phase=source,
                                 to_phase=target, event=event, time_s=12.5, time=12.5)
    assert isinstance(error, ModelDomainExceeded)
    assert not hasattr(error, 'constraints')
    assert failure_details(error)['kind'] == 'unsupported_phase_change'
    assert not failure_details(error)['fatal']


def test_phase_errors_are_unresolved_and_do_not_trip_failure_monitor():
    error = UnsupportedPhaseChangeError('supply', 'Nitrogen', 'gas', 'saturated', 'condense', 48.)
    result = finalize(SimResult(termination='unsupported_phase_change'),
                      {'goal_apogee': 150000., 'max_burn_duration': 75.})
    assert result.feasible is None and not result.accepted
    assert candidate_class(result) == 'unresolved'
    score, violations = candidate_score(result, dict(constraint_scales={}, unit_scales={'1': 1.}))
    assert score == 3.
    assert violations == {}
    assert not any('phase_change' in key for key in result.constraint_records)
    monitor = FailureMonitor(dict(consecutive_limit=1, window=1, rate_limit=.01, repeated_limit=1))
    for _ in range(100):
        monitor.observe(dict(termination='unsupported_phase_change', failure_details=failure_details(error)))
    assert monitor.consecutive == 0


def test_phase_change_during_network_initialization_is_a_model_domain_error():
    from Fluids.FluidNetwork import FluidNetwork
    from FluidTables.PropertyModels import SaturationProperties
    from test_fluid_network import Properties, gas_tank

    class Saturating(Properties):
        def supports_saturation(self, fluid):
            return True

        def saturation_bounds(self, fluid):
            return (1., 1e6)

        def saturation_at_p(self, fluid, pressure):
            return SaturationProperties(pressure, 300.,
                                        self.state_pt('liquid', pressure, 300.),
                                        self.state_pt('gas', pressure, 300.))

    tank = gas_tank(3e5, .01)
    tank['state0']['T'] = 290.
    tank['state0']['P'] *= 290 / 300
    tank['state0']['U'] *= 290 / 300
    with FluidNetwork({'other_supply': tank}, {}, fluid_properties=Saturating()) as net:
        with pytest.raises(UnsupportedPhaseChangeError) as failure:
            net.initialize()
        assert failure.value.details["time_s"] == 0.
        assert failure.value.details['node_id'] == 'other_supply'
        assert failure.value.details['event'] == 'condense'
        assert net.nodes['other_supply'].mode == 'gas'
        assert not net.events


@pytest.mark.parametrize('phase', ['sizing', 'flight initialization', 'runtime'])
def test_optimizer_records_phase_error_and_continues_without_constraint(tmp_path, monkeypatch, phase):
    import json
    import time
    from optimizer.core import Evaluator

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.settings = dict(max_evaluations=2, constraint_scales={}, unit_scales={})
    evaluator.paths = {}
    evaluator.count = 0
    evaluator.best = None
    evaluator.population_size = None
    evaluator.started = time.perf_counter()
    evaluator.output = tmp_path
    evaluator.failure_monitor = FailureMonitor(dict(consecutive_limit=1, window=1, rate_limit=.01))
    monkeypatch.setattr(evaluator, 'decode', lambda values: {})

    def fail(cfg):
        try:
            raise UnsupportedPhaseChangeError('arbitrary_volume', 'Nitrogen', 'gas',
                                              'saturated', 'condense', 48.)
        except UnsupportedPhaseChangeError as error:
            if phase == 'sizing':
                raise
            raise EvaluationFailure(phase, cfg, SimResult(final_time=46.)) from error

    monkeypatch.setattr(evaluator, 'evaluate', fail)
    assert evaluator([]) == 3.
    assert evaluator([]) == 3.
    rows = [json.loads(line) for line in (tmp_path/'evaluations.jsonl').read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row['score_class'] == 'unresolved'
        assert not row['accepted']
        assert row['termination'] == 'unsupported_phase_change'
        assert row['constraints'] == {} and row['violations'] == {}
        assert row['failure_details']['details']['node_id'] == 'arbitrary_volume'
        assert row['failure_details']['details']['time_s'] == 48.
        assert not row['failure_details']['fatal']
        assert (tmp_path/f"failure_{row['evaluation']:04d}.txt").exists()
    assert evaluator.failure_monitor.consecutive == 0
