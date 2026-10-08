import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from diagnostics.errors import (LookupBoundsError, ModelDomainExceeded, SolverConvergenceError,
                    SolverSetupError, ResidualAcceptanceError, InfrastructureError,
                    SearchFailureLimit, failure_details)
from Fluids.Sundials.ida_session import IdaSession
from optimizer.core import FailureMonitor
import optimizer.workers as search


def session():
    item = IdaSession.__new__(IdaSession)
    item.callback_error = item.last_trial_error = None
    item.time = 1.
    return item


@pytest.mark.parametrize('status,retryable', [(-1, False), (-2, False), (-3, True), (-4, True), (-13, True)])
def test_convergence_status_is_typed(status, retryable):
    with pytest.raises(SolverConvergenceError) as failure:
        session()._check(status, 'integration')
    assert failure.value.retryable is retryable
    assert failure.value.details['status'] == status


@pytest.mark.parametrize('status,operation', [(-22, 'integration'), (-1, 'SUNContext_Create'), (-5, 'integration')])
def test_setup_failures_are_not_nonconvergence(status, operation):
    with pytest.raises(SolverSetupError):
        session()._check(status, operation)


def test_bounds_recoverable_only_in_residual_callback():
    item = session()
    error = LookupBoundsError('outside', table='engine', axis='Pc', value=1., lower=2., upper=3.)
    def residual(*args):
        raise error
    item.residual = item.roots = residual
    item.core = SimpleNamespace(N_VGetNumpyArray=lambda x: x)
    item.callback_calls = item.recoverable_errors = 0
    assert item._residual_callback(0, np.zeros(1), np.zeros(1), np.zeros(1), None) == 1
    with pytest.raises(ModelDomainExceeded) as failure:
        item._check(-9, 'integration')
    assert failure.value.__cause__ is error
    details = failure_details(failure.value)
    assert details['kind'] == 'table_domain_exceeded'
    assert details['details']['axis'] == 'Pc'
    assert item._root_callback(0, None, None, np.zeros(1), None) == -1


def test_programming_callback_error_preserves_type():
    item = session()
    item.callback_error = TypeError('bad callback')
    with pytest.raises(TypeError):
        item._check(-8, 'integration')


def test_recovered_lookup_trial_does_not_mask_later_nonconvergence():
    item = session()
    item.last_trial_error = LookupBoundsError('previous trial was out of bounds')
    item.core = SimpleNamespace(N_VGetNumpyArray=lambda x: x)
    item.residual = lambda t, y, dy, out: out.fill(0.)
    item.callback_calls = 0
    assert item._residual_callback(0, np.zeros(1), np.zeros(1), np.zeros(1), None) == 0
    with pytest.raises(SolverConvergenceError):
        item._check(-4, 'integration')


@pytest.mark.parametrize('error,kind,fatal', [
    (ResidualAcceptanceError('closure'), 'residual_acceptance_failure', False),
    (RuntimeError('unknown'), 'unexpected_error', False),
    (ValueError('bad config'), 'configuration_error', True),
    (PermissionError('locked'), 'infrastructure_error', True),
])
def test_classification(error, kind, fatal):
    assert failure_details(error)['kind'] == kind
    assert failure_details(error)['fatal'] is fatal


def row(kind):
    return {'failure_details': {'kind': kind, 'fingerprint': kind}}


def test_monitor_excludes_physical_and_table_rejections():
    monitor = FailureMonitor()
    for _ in range(100):
        monitor.observe(row('table_domain_exceeded'))
        monitor.observe(row('infeasible_operating_state'))
    for _ in range(4):
        monitor.observe(row('solver_nonconvergence'))
    with pytest.raises(SearchFailureLimit, match='consecutive'):
        monitor.observe(row('solver_nonconvergence'))


def test_monitor_repeated_unknown_and_window_rate():
    monitor = FailureMonitor()
    for _ in range(2):
        monitor.observe(row('unexpected_error'))
        monitor.observe({})
    with pytest.raises(SearchFailureLimit, match='Repeated'):
        monitor.observe(row('unexpected_error'))
    monitor = FailureMonitor(dict(window=4, rate_limit=.25))
    for kind in ('solver_nonconvergence', 'success', 'success'):
        monitor.observe(row(kind))
    with pytest.raises(SearchFailureLimit, match='rate'):
        monitor.observe(row('residual_acceptance_failure'))


@pytest.mark.parametrize('code,kind', [(3221225477, 'native_crash'), (-1073741819, 'native_crash'),
                                      (-11, 'native_crash'), (1, 'worker_error')])
def test_native_exit_classification(code, kind):
    assert search.exit_kind(code) == kind


def test_atomic_publication_retries_without_losing_old_result(tmp_path, monkeypatch):
    destination = tmp_path/'result.json'
    destination.write_text('{"old": true}')
    original, calls = Path.replace, []
    def replace(path, target):
        calls.append(path)
        assert json.loads(destination.read_text()) == {'old': True}
        if len(calls) < 3:
            raise PermissionError('reader holds destination')
        return original(path, target)
    monkeypatch.setattr(Path, 'replace', replace)
    assert search.write_json(destination, {'new': True})
    assert len(calls) == 3
    assert json.loads(destination.read_text()) == {'new': True}
    assert not list(tmp_path.glob('*.tmp'))


def test_persistent_publication_error_is_fatal_unless_optional(tmp_path, monkeypatch, caplog):
    def replace(*args):
        raise PermissionError('still locked')
    monkeypatch.setattr(Path, 'replace', replace)
    with pytest.raises(InfrastructureError):
        search.write_json(tmp_path/'result.json', {}, retry_seconds=0)
    assert not search.write_json(tmp_path/'status.json', {}, optional=True, retry_seconds=0)
    assert 'progress_publication_failed' in caplog.text


def test_fatal_worker_marker_never_becomes_candidate_penalty(tmp_path):
    job = tmp_path/'candidates'/'0001'
    job.mkdir(parents=True)
    (job/'fatal.json').write_text('{}')
    with pytest.raises(InfrastructureError):
        search.launch_batch(tmp_path, [(1, [0.])], 1.)


@pytest.mark.parametrize('kind,expected_calls', [('residual', 3), ('convergence', 3), ('configuration', 1)])
def test_retry_restores_state_and_preserves_error_type(kind, expected_calls):
    from Fluids.FluidNetwork import FluidNetwork
    net = FluidNetwork.__new__(FluidNetwork)
    net.time, net.effective_rtol, net.effective_max_step = 0., 1e-4, 1.
    net.options = dict(residual_retries=2, retry_rtol_floor=1e-8, retry_rtol_factor=.1)
    net.retry_diagnostics = []
    net._checkpoint = lambda: None
    def restore(checkpoint):
        net.time, net.effective_rtol, net.effective_max_step = 0., 1e-4, 1.
    net.restore = restore
    calls = []
    error = {'residual': ResidualAcceptanceError('closure'),
             'convergence': SolverConvergenceError('integration', -4, 0.),
             'configuration': ValueError('bad input')}[kind]
    def advance(*args):
        assert net.time == 0.
        calls.append((net.effective_rtol, net.effective_max_step))
        net.time = .5
        raise error
    net._advance_interval = advance
    with pytest.raises(type(error)) as failure:
        net.update(1.)
    assert failure.value is error
    assert len(calls) == expected_calls
    assert net.time == 0.
    assert len(error.attempts) == expected_calls
    if kind == 'convergence':
        assert all(rtol == 1e-4 for rtol, _ in calls)
        assert calls[-1][1] < calls[0][1]
    elif kind == 'residual':
        assert calls[-1][0] == pytest.approx(1e-6)


def test_native_retry_uses_fresh_process_and_retains_attempt(tmp_path, monkeypatch):
    stub = tmp_path/'worker.py'
    stub.write_text('''
import argparse, json, os
p=argparse.ArgumentParser()
p.add_argument('--output')
p.add_argument('--worker-batch', nargs='+', type=int)
a=p.parse_args()
from pathlib import Path
for index in a.worker_batch:
    job=Path(a.output)/'candidates'/f'{index:04d}'
    if not (job/'failed_once').exists():
        (job/'failed_once').write_text(str(os.getpid()))
        os._exit(99)
    task=json.loads((job/'task.json').read_text())
    row=dict(index=index, coordinates=task['coordinates'], pid=os.getpid(),
             score=1.5, score_class='completed_infeasible')
    (job/'result.json').write_text(json.dumps(row))
''')
    popen = search.subprocess.Popen
    def launch_stub(command, **kwargs):
        assert command[:3] == [search.sys.executable, '-m', 'optimizer.workers']
        return popen([command[0], str(stub), *command[3:]], **kwargs)
    monkeypatch.setattr(search.subprocess, 'Popen', launch_stub)
    monkeypatch.setattr(search, 'exit_kind', lambda code: 'native_crash' if code == 99 else 'worker_error')
    (tmp_path/'search.yaml').write_text('failure_policy:\n  native_retries: 1\n')
    rows = search.launch_batch(tmp_path, [(1, [0.])], 5.)
    job = tmp_path/'candidates'/'0001'
    assert rows[0]['score'] == 1.5
    assert rows[0]['worker_attempts'][0]['kind'] == 'native_crash'
    assert rows[0]['pid'] != int((job/'failed_once').read_text())
    assert (job/'native_attempt_1.stderr.log').exists()
