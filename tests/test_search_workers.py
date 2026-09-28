"""Worker deadlines and recycling survive actual subprocess exits."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples import run_optimizer_search as search


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
    search.main(SimpleNamespace(output=tmp_path, workers=2, candidates_per_worker=3, timeout=1.))
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
    monkeypatch.setattr(search, '__file__', str(stub))
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
