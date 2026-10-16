import pytest

import main
from copy import deepcopy
from simulation_types import SimResult


def test_manual_run_passes_config_to_shared_simulate_unchanged(monkeypatch):
    cfg = {'constraints': {'max_length_to_diameter': 25., 'goal_apogee': 150000.},
           'tanks': {'press_tank': {'min_temperature': 190.}}, 'simulation': {}}
    before = deepcopy(cfg)
    monkeypatch.setattr('sys.argv', ['main.py', 'candidate.yaml'])
    monkeypatch.setattr(main, 'load_config', lambda path: cfg)
    def simulate(candidate, **kwargs):
        assert candidate == before
        assert 'assess_constraints' not in kwargs
        raise RuntimeError('checked manual policy')
    monkeypatch.setattr(main, 'simulate', simulate)
    with pytest.raises(RuntimeError, match='checked manual policy'):
        main.main()
    assert cfg == before


def test_cli_reports_typed_failure_without_writing_partial_outputs(monkeypatch, tmp_path):
    from diagnostics.errors import EvaluationFailure, LookupBoundsError
    monkeypatch.setattr('sys.argv', ['main.py', 'candidate.yaml'])
    monkeypatch.setattr(main, 'load_config', lambda path: {'simulation': {'output': str(tmp_path / 'flight.csv')}})
    def fail(*args, **kwargs):
        raise EvaluationFailure('runtime', {}, warnings=[dict(code='test', message='caution', count=1)]) from LookupBoundsError('AoA beyond table')
    monkeypatch.setattr(main, 'simulate', fail)
    monkeypatch.setattr(main, 'write_history', lambda *a: pytest.fail('Failed run wrote history'))
    with pytest.raises(SystemExit) as caught:
        main.main()
    assert 'LookupBoundsError' in str(caught.value) and 'AoA beyond table' in str(caught.value)
    assert 'Warning [test]' in str(caught.value)
    assert (tmp_path / 'flight.failure.json').exists()


def test_main_dispatches_sweep(monkeypatch):
    from unittest.mock import Mock
    import sensitivity
    runner = Mock()
    monkeypatch.setattr('sys.argv', ['main.py', '--sweep', 'sweep.yaml', '--dry-run'])
    monkeypatch.setattr(sensitivity, 'run_file', runner)
    main.main()
    assert runner.call_args.kwargs == {'dry_run': True}


def test_cli_keeps_unexplained_empty_history_as_an_error(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "candidate.yaml"])
    monkeypatch.setattr(main, "load_config", lambda path: {})
    monkeypatch.setattr(main, "simulate", lambda *args, **kwargs: SimResult())
    with pytest.raises(RuntimeError, match="no time steps.*time_limit"):
        main.main()
