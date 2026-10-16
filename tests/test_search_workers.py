"""Worker deadlines and recycling survive actual subprocess exits."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

import optimizer.workers as search
import optimizer.core as optimizer


@pytest.mark.parametrize('budget,expected,generations', [(100, 32, 3), (11, 11, 0)])
def test_parallel_search_honors_budgets_with_flat_scores(tmp_path, monkeypatch, budget, expected, generations):
    # Exercise the real DE driver and map/callback, without expensive simulations.
    settings = dict(seed=42, maxiter=3, max_evaluations=budget, popsize=4,
                    tank_ids={}, bounds={'a': [0., 1.], 'b': [0., 1.]})
    (tmp_path/'base.yaml').write_text('{}')
    (tmp_path/'search.yaml').write_text(search.yaml.safe_dump(settings))
    (tmp_path/'fingerprints.json').write_text('{}')
    monkeypatch.setattr(search, 'fingerprints', lambda: {})
    monkeypatch.setattr(search.opt, 'variable_paths', lambda *args: {'a': 'a', 'b': 'b'})
    monkeypatch.setattr(search, 'launch_batch', lambda output, tasks, timeout, **kwargs: [
        dict(index=i, score=3., score_class='unresolved', accepted=False,
             termination='test_rejection', wall_seconds=0.) for i, _ in tasks])
    search.optimize_parallel(None, tmp_path, workers=2, candidates_per_worker=3, timeout=1., resume=True)
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert summary['evaluations'] == expected
    assert summary['generations'] == generations
    assert summary['stop'] == ('evaluation budget exhausted' if budget == 11 else 'generation budget exhausted')


def test_changed_sources_reject_search_resumption(tmp_path, monkeypatch):
    monkeypatch.setattr(search, 'fingerprints', lambda: {'solver.py': 'first-version'})
    (tmp_path/'fingerprints.json').write_text(json.dumps(search.fingerprints()))
    first = search.check_sources(tmp_path)
    assert first == search.check_sources(tmp_path)
    monkeypatch.setattr(search, 'fingerprints', lambda: {'solver.py': 'edited-version'})
    with pytest.raises(RuntimeError, match='sources changed'):
        search.check_sources(tmp_path)


@pytest.mark.parametrize('failure', ['crash', 'timeout', 'reported'])
def test_batch_recycles_after_failure_and_resumes_completed_jobs(tmp_path, monkeypatch, failure):
    stub = tmp_path/'worker.py'
    stub.write_text('''
import argparse, json, os, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output', type=Path)
p.add_argument('--worker-batch', nargs='+', type=int)
a=p.parse_args()
for index in a.worker_batch:
    job=a.output/'candidates'/f'{index:04d}'
    task=json.loads((job/'task.json').read_text())
    mode=json.loads((a.output/'mode.json').read_text())
    if index == 2 and mode == 'crash': os._exit(99)
    if index == 2 and mode == 'timeout': time.sleep(10)
    row=dict(index=index, coordinates=task['coordinates'], pid=os.getpid(),
             score_class='unresolved' if index==2 else 'completed_infeasible')
    (job/'result.json').write_text(json.dumps(row))
    if index==2: break
''')
    popen = search.subprocess.Popen
    def launch_stub(command, **kwargs):
        assert command[:3] == [search.sys.executable, '-m', 'optimizer.workers']
        return popen([command[0], str(stub), *command[3:]], **kwargs)
    monkeypatch.setattr(search.subprocess, 'Popen', launch_stub)
    (tmp_path/'mode.json').write_text(json.dumps(failure))
    tasks = [(i, [float(i)]) for i in range(1, 5)]
    rows = search.launch_batch(tmp_path, tasks, timeout=1.)
    assert [r['index'] for r in rows] == [1, 2, 3, 4]
    assert rows[1]['score_class'] == 'unresolved'
    assert rows[2]['pid'] == rows[3]['pid'] != rows[0]['pid']
    if failure != 'reported':
        assert rows[1]['termination'] == ('wall_timeout' if failure == 'timeout' else 'worker_error')
    # Resuming preserves results and their coordinate association.
    assert search.launch_batch(tmp_path, tasks, timeout=1.) == rows
    with pytest.raises(AssertionError):
        search.launch_batch(tmp_path, [(1, [-1.])], timeout=1.)


@pytest.mark.parametrize('verification', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_worker_uses_shared_evaluation_and_simulation_cleanup(tmp_path, monkeypatch, verification, fail):
    from types import SimpleNamespace
    import numpy as np
    import simulation
    from diagnostics.constraints import finalize
    from diagnostics.errors import SolverConvergenceError
    from simulation_types import SimResult

    settings = optimizer.load_config('Configs/optimizer_epump.yaml')
    settings.update(conditional_geometry=False, profile=False, verification=verification)
    settings.pop('soft_penalties', None)
    cfg = optimizer.load_config(settings['base_config'])
    cfg['tanks']['press_tank']['volume'] = .02
    (tmp_path/'base.yaml').write_text(optimizer.yaml.safe_dump(cfg))
    (tmp_path/'search.yaml').write_text(optimizer.yaml.safe_dump(settings))
    evaluator = optimizer.Evaluator(cfg, settings, None, None, None, tmp_path)
    coordinates = [] if verification else np.mean(evaluator.bounds, axis=1).tolist()
    for index in (1, 2):
        job = tmp_path/'candidates'/f'{index:04d}'
        job.mkdir(parents=True)
        (job/'task.json').write_text(json.dumps(dict(coordinates=coordinates)))

    closed, runs, decoded, models = [], [], [], []
    def sources(cfg):
        models.append(True)
        return None, None
    monkeypatch.setattr(optimizer, 'property_sources', sources)
    monkeypatch.setattr(optimizer, 'DragModel', lambda _: None)
    original_decode = optimizer.Evaluator.decode
    def decode(self, values):
        assert not verification
        decoded.append(values)
        return original_decode(self, values)
    monkeypatch.setattr(optimizer.Evaluator, 'decode', decode)

    def run(candidate, *, context, progress, compute_loads, **kwargs):
        runs.append(candidate)
        assert compute_loads is verification
        result = finalize(SimResult(apogee=150000, initial_mass=500, burn_complete=True,
                                    burn_duration=40, load_peaks={'force': 42} if compute_loads else {}),
                          {'goal_apogee': 150000, 'max_burn_duration': 45})
        context.update(phase='runtime', flight=SimpleNamespace(result=result),
                       propulsion=SimpleNamespace(network=SimpleNamespace(retry_diagnostics=[{'rtol': 1e-6}],
                                                  pressure_tracking=SimpleNamespace(records={})),
                                                  close=lambda: closed.append(True)))
        progress(SimpleNamespace(t=0., h=0.))
        if fail:
            raise SolverConvergenceError('integration', -4, 1.)
        return result
    monkeypatch.setattr(simulation, '_simulate', run)
    simulate = optimizer.simulate
    propulsion_type = simulation.PropSystem
    search.worker(tmp_path, [1, 2])

    count = 1 if fail else 2
    assert len(runs) == len(closed) == count
    assert len(decoded) == (0 if verification else count)
    assert models == [True]
    assert optimizer.simulate is simulate and simulation.PropSystem is propulsion_type
    for index in range(1, count + 1):
        job = tmp_path/'candidates'/f'{index:04d}'
        assert (job/'progress.json').exists()
        assert not (job/'evaluations.jsonl').exists()
        assert not (job/'best.yaml').exists()
        saved = optimizer.load_config(job/'candidate.yaml')
        assert saved == runs[index - 1]
        assert saved['simulation']['dt'] == cfg['simulation']['dt'] * (settings['verification_dt_factor'] if verification else 1)
        if fail and verification:
            assert (job/'fatal.json').exists()
            assert not (job/'result.json').exists()
        else:
            row = json.loads((job/'result.json').read_text())
            assert row['score_class'] == ('unresolved' if fail else 'feasible')
            assert row['recovered_fluid_retries'] == [{'rtol': 1e-6}]
            assert row['load_peaks'] == ({'force': 42} if verification else {})
            assert row['model_cache_hit'] is (index == 2)
    if fail:
        assert not (tmp_path/'candidates'/'0002'/'result.json').exists()


@pytest.mark.parametrize('passes', [False, True])
def test_parallel_verifies_saved_config_without_synthetic_bounds(tmp_path, monkeypatch, passes):
    from types import SimpleNamespace
    from diagnostics.constraints import finalize
    from simulation_types import SimResult

    settings = optimizer.load_config('Configs/optimizer_epump.yaml')
    settings.update(conditional_geometry=False, max_evaluations=1)
    settings.pop('soft_penalties', None)
    calls = []
    def run(cfg, *, compute_loads, progress, **kwargs):
        calls.append((deepcopy(cfg), compute_loads))
        progress(SimpleNamespace(t=0., h=0.))
        return finalize(SimResult(apogee=150000 if not compute_loads or passes else 140000,
                                  initial_mass=500, burn_complete=True, burn_duration=40),
                        {'goal_apogee': 150000, 'max_burn_duration': 45})
    monkeypatch.setattr(optimizer, 'simulate', run)
    monkeypatch.setattr(optimizer, 'property_sources', lambda _: (None, None))
    def launch(output, tasks, timeout, **kwargs):
        for index, coordinates in tasks:
            job = output/'candidates'/f'{index:04d}'
            job.mkdir(parents=True)
            (job/'task.json').write_text(json.dumps(dict(coordinates=list(map(float, coordinates)))))
        search.worker(output, [index for index, _ in tasks])
        return [json.loads((output/'candidates'/f'{index:04d}'/'result.json').read_text()) for index, _ in tasks]
    monkeypatch.setattr(search, 'launch_batch', launch)
    output = tmp_path/'run'
    summary = optimizer.optimize(settings, output, workers=2)
    assert summary['verified'] is passes
    assert summary['evaluations'] == 1
    assert len(calls) == 2
    search_cfg, search_loads = calls[0]
    verify_cfg, verify_loads = calls[1]
    assert search_loads is False and verify_loads is True
    expected = deepcopy(search_cfg)
    expected['simulation']['dt'] *= settings['verification_dt_factor']
    assert verify_cfg == expected
    assert optimizer.load_config(output/'best.yaml') == search_cfg
    assert optimizer.load_config(output/'verification'/'search.yaml')['bounds'] == settings['bounds']
    assert json.loads((output/'verification.json').read_text())['search_coordinates'] == []
    assert (output/'verified.yaml').exists() is passes
    if passes:
        assert optimizer.load_config(output/'verified.yaml') == verify_cfg
