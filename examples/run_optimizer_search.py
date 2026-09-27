"""Resumable diagnostic DE search, with bounded, recyclable candidate workers.

Solver fixes belong in OUTPUT/diagnostic_fixes.py, whose install() function is
called only inside these diagnostic workers. Main solver files are fingerprinted.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import gc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import faulthandler
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
import optimizer as opt


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def fingerprints():
    files = [ROOT/'optimizer.py', ROOT/'simulation.py', ROOT/'constraints.py', ROOT/'simulation_types.py']
    files.extend((ROOT/'examples'/name for name in ('run_optimizer_search.py', 'search_timing.py')))
    for folder in ('Fluids', 'Flight', 'FluidTables', 'Vehicle', 'Configs'):
        files.extend((ROOT/folder).rglob('*.py'))
        files.extend((ROOT/folder).rglob('*.yaml'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def check_sources(output):
    saved = output/'fingerprints.json'
    current = fingerprints()
    if saved.exists() and json.loads(saved.read_text()) != current:
        raise RuntimeError('Main sources changed during this search; use a new output directory')
    return hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()


def worker(output, indices):
    import simulation
    from Fluids.PropSystem import PropSystem
    import CoolProp.CoolProp as CP
    tracked = []
    class Tracked(PropSystem):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tracked.append(self)
    opt.PropSystem = simulation.PropSystem = Tracked
    def no_live_properties(*args, **kwargs):
        raise AssertionError('Live CoolProp call in a table-only sweep')
    CP.PropsSI = no_live_properties
    patch = output/'diagnostic_fixes.py'
    patch_version = hashlib.sha256(patch.read_bytes()).hexdigest() if patch.exists() else 'none'
    if patch.exists():
        spec = importlib.util.spec_from_file_location('diagnostic_fixes', patch)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install()
    models, timing = None, None
    for index in indices:
        source_version = check_sources(output)
        job = output/'candidates'/f'{index:04d}'
        task = json.loads((job/'task.json').read_text())
        faulthandler.enable()
        faulthandler.dump_traceback_later(240, repeat=False)
        started, cpu_started = time.perf_counter(), time.process_time()
        original_simulate = opt.simulate
        try:
            cfg, settings = opt.load_config(output/'base.yaml'), opt.load_config(output/'search.yaml')
            profiled = settings.get('profile', index % 16 == 0)
            if profiled and timing is None:
                from search_timing import install
                timing = install()
            if timing is not None:
                timing.enabled = profiled
                timing.records.clear()
            loading = time.process_time()
            reused_models = models is not None
            if models is None:
                model = opt.DragModel(opt.project_path(cfg['aero']['model']))
                pure, combustion = opt.property_sources(cfg)
                models = model, pure, combustion
            model, pure, combustion = models
            loading_cpu = time.process_time() - loading
            evaluator = opt.Evaluator(cfg, settings, model, pure, combustion, job)
            last_progress = [0.]
            def progress(kin):
                now = time.perf_counter()
                if now - last_progress[0] >= 10 or kin.t == 0:
                    write_json(job/'progress.json', dict(time_s=kin.t, altitude=kin.h,
                               wall_seconds=now-started))
                    last_progress[0] = now
                    if timing is not None and timing.enabled:
                        timing.save(job/'timings.json')
            simulation_results = []
            def monitored(*args, **kwargs):
                if settings.get('verification', False):
                    kwargs['compute_loads'] = True
                result = original_simulate(*args, **kwargs, progress=progress)
                simulation_results.append(result)
                return result
            opt.simulate = monitored
            try:
                candidate = evaluator.decode(task['coordinates'])
                (job/'candidate.yaml').write_text(yaml.safe_dump(candidate, sort_keys=False))
            except opt.GeometryError:
                pass
            score = evaluator(task['coordinates'])
            row = json.loads((job/'evaluations.jsonl').read_text().splitlines()[-1])
            if check_sources(output) != source_version:
                raise RuntimeError('Sources changed while evaluating this candidate; discard this evaluation')
            row.update(index=index, coordinates=task['coordinates'], patch_version=patch_version, source_version=source_version,
                       wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu_started,
                       profiled=profiled, model_cache_hit=reused_models, loading_cpu_seconds=loading_cpu,
                       timing_version=(hashlib.sha256((ROOT/'examples/search_timing.py').read_bytes()).hexdigest()
                                       if profiled else None),
                       recovered_fluid_retries=[d for p in tracked for d in p.network.retry_diagnostics],
                       load_peaks=simulation_results[-1].load_peaks if simulation_results else {})
            write_json(job/'result.json', row)
            # Recycle after any unresolved solve; never reuse a potentially damaged native session.
            if row['score_class'] == 'unresolved':
                return
        finally:
            if timing is not None and timing.enabled:
                timing.save(job/'timings.json')
            faulthandler.cancel_dump_traceback_later()
            opt.simulate = original_simulate
            for prop in tracked:
                prop.close()
            tracked.clear()
            gc.collect()


def launch_batch(output, tasks, timeout):
    """Supervise a small batch, enforcing a separate wall deadline per candidate.

    Successful jobs share only loaded models. A timeout, native exit or reported
    numerical failure ends that child; remaining jobs start in a clean process.
    """
    pending = []
    for index, coordinates in tasks:
        job = output/'candidates'/f'{index:04d}'
        job.mkdir(parents=True, exist_ok=True)
        task = dict(index=index, coordinates=list(map(float, coordinates)))
        if (job/'result.json').exists():
            row = json.loads((job/'result.json').read_text())
            np.testing.assert_allclose(row['coordinates'], coordinates, rtol=1e-13, atol=1e-13)
        else:
            write_json(job/'task.json', task)
            pending.append(index)
    while pending:
        first = output/'candidates'/f'{pending[0]:04d}'
        current, started = pending[0], time.perf_counter()
        status = 'worker_error'
        with (first/'stdout.log').open('w') as stdout, (first/'stderr.log').open('w') as stderr:
            with subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    '--output', str(output), '--worker-batch', *map(str, pending)],
                    cwd=ROOT, stdout=stdout, stderr=stderr) as process:
                while True:
                    remaining = [i for i in pending if not (output/'candidates'/f'{i:04d}'/'result.json').exists()]
                    if remaining and remaining[0] != current:
                        current, started = remaining[0], time.perf_counter()
                    elapsed = time.perf_counter()-started
                    if elapsed >= timeout:
                        status = 'wall_timeout'
                        process.kill()
                        process.wait()
                        break
                    try:
                        process.wait(timeout=min(.25, timeout-elapsed))
                        status = 'native_crash' if process.returncode < 0 else 'worker_error'
                        break
                    except subprocess.TimeoutExpired:
                        pass
        remaining = [i for i in pending if not (output/'candidates'/f'{i:04d}'/'result.json').exists()]
        completed = [i for i in pending if i not in remaining]
        reported_failure = bool(completed) and json.loads(
            (output/'candidates'/f'{completed[-1]:04d}'/'result.json').read_text())['score_class'] == 'unresolved'
        if remaining and not reported_failure:
            index = remaining.pop(0)
            job = output/'candidates'/f'{index:04d}'
            task = json.loads((job/'task.json').read_text())
            row = dict(index=index, coordinates=task['coordinates'], score=3., score_class='unresolved',
                       accepted=False, termination=status, mass=None, apogee=None, burn_duration=None,
                       wall_seconds=time.perf_counter()-started, violations={},
                       failure=(first/'stderr.log').read_text()[-18000:])
            write_json(job/'result.json', row)
        pending = remaining
    return [json.loads((output/'candidates'/f'{i:04d}'/'result.json').read_text()) for i, _ in tasks]


def launch(output, index, coordinates, timeout):
    return launch_batch(output, [(index, coordinates)], timeout)[0]


def main(args):
    output = args.output.resolve()
    if not output.exists():
        settings = opt.load_config(args.config)
        model = opt.DragModel(opt.project_path(opt.load_config(settings['base_config'])['aero']['model']))
        cfg, settings = opt.prepare(settings, model)
        output.mkdir(parents=True)
        (output/'base.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        (output/'search.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
        write_json(output/'fingerprints.json', fingerprints())
        (output/'diagnostic_fixes.py').write_text('"""Temporary solver experiments only; initially no patches."""\n\ndef install():\n    pass\n')
    else:
        if json.loads((output/'fingerprints.json').read_text()) != fingerprints():
            raise RuntimeError('Main sources changed since this search started; use a new output directory')
        cfg, settings = opt.load_config(output/'base.yaml'), opt.load_config(output/'search.yaml')
    names = list(opt.variable_paths(cfg, settings['tank_ids']))
    bounds = [settings['bounds'][name] for name in names]
    budget, count = settings['max_evaluations'], 0
    rows = {}
    started = time.perf_counter()
    def mapped(function, vectors):
        nonlocal count
        check_sources(output)
        vectors = list(vectors)
        available = min(len(vectors), budget-count)
        tasks = [(count+i+1, vector) for i, vector in enumerate(vectors[:available])]
        batches = [tasks[i:i+args.candidates_per_worker] for i in range(0, len(tasks), args.candidates_per_worker)]
        futures = [pool.submit(launch_batch, output, batch, args.timeout) for batch in batches]
        for future in as_completed(futures):
            for row in future.result():
                rows[row['index']] = row
                counts = Counter(r['score_class'] for r in rows.values())
                write_json(output/'status.json', dict(completed=len(rows), budget=budget, classes=dict(counts),
                           best_score=min(r['score'] for r in rows.values()), latest_index=row['index']))
                print(f"{row['index']}/{budget}: {row['score_class']} score={row['score']:.7g} "
                      f"apogee={row.get('apogee')} wall={row['wall_seconds']:.1f}s", flush=True)
        count += available
        if available < len(vectors):
            raise opt.EvaluationBudget
        return [rows[i]['score'] for i, _ in tasks]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def generation_finished(intermediate_result):
            nonlocal generation
            generation = intermediate_result.nit
            with (output/'generations.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(generation=intermediate_result.nit, evaluations=count,
                    best_score=float(intermediate_result.population_energies.min()),
                    median_score=float(np.median(intermediate_result.population_energies))))+'\n')

        with opt.GenerationBudgetSolver(lambda x: 0, bounds, rng=np.random.RandomState(settings['seed']),
                popsize=settings['popsize'], maxiter=settings['maxiter'], polish=False,
                workers=mapped, updating='deferred', callback=generation_finished) as solver:
            generation = 0
            try:
                result = solver.solve()
                generation = result.nit
            except opt.EvaluationBudget:
                pass
    ordered = [rows[i] for i in sorted(rows)]
    (output/'evaluations.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in ordered))
    best = min(ordered, key=lambda r: r['score'])
    accepted = [r for r in ordered if r['accepted']]
    verified = None
    if accepted:
        winner = min(accepted, key=lambda r: r['mass'])
        (output/'best.yaml').write_text((output/'candidates'/f"{winner['index']:04d}"/'candidate.yaml').read_text())
        write_json(output/'best.json', winner)
        verification = output/'verification'
        verification.mkdir(exist_ok=True)
        verify_cfg = opt.load_config(output/'best.yaml')
        verify_cfg['simulation']['dt'] *= settings['verification_dt_factor']
        verify_settings = dict(settings, conditional_geometry=False, verification=True, max_evaluations=1)
        paths = opt.variable_paths(verify_cfg, settings['tank_ids'])
        verify_values = [float(opt.get_path(verify_cfg, path)) for path in paths.values()]
        verify_settings['bounds'] = {name: [value, value] for name, value in zip(paths, verify_values)}
        (verification/'base.yaml').write_text(yaml.safe_dump(verify_cfg, sort_keys=False))
        (verification/'search.yaml').write_text(yaml.safe_dump(verify_settings, sort_keys=False))
        shutil.copyfile(output/'diagnostic_fixes.py', verification/'diagnostic_fixes.py')
        verification_result = launch(verification, 1, verify_values, 2*args.timeout)
        write_json(output/'verification.json', verification_result)
        verified = verification_result['accepted']
        if verified:
            (output/'verified.yaml').write_text(yaml.safe_dump(verify_cfg, sort_keys=False))
    summary = dict(evaluations=len(ordered), counts=dict(Counter(r['termination'] for r in ordered)),
                   generations=generation,
                   stop='evaluation budget exhausted' if count >= budget else 'generation budget exhausted',
                   classes=dict(Counter(r['score_class'] for r in ordered)), accepted=len(accepted),
                   best_index=best['index'], best_score=best['score'], verified=verified,
                   elapsed_this_invocation=time.perf_counter()-started,
                   candidate_seconds=sum(r['wall_seconds'] for r in ordered))
    write_json(output/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='Configs/optimizer_pressure_fed.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=300.)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--worker-batch', type=int, nargs='+')
    parser.add_argument('--candidates-per-worker', type=int, default=3,
                        help='Recycle each worker after this many candidates; 1 disables model reuse')
    parser.add_argument('--patch', type=Path, help='Temporary patch module for a replay only')
    parser.add_argument('--profile', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable/disable timing for a diagnostic replay; search samples every 16th candidate')
    parser.add_argument('--replay', type=int, help='Re-evaluate an index with output/diagnostic_fixes.py; leave search scores intact')
    args = parser.parse_args()
    if args.candidates_per_worker < 1 or args.workers < 1 or not 0 < args.timeout < float('inf'):
        parser.error('workers, candidates-per-worker and timeout must be positive and finite')
    if args.worker_batch is not None:
        worker(args.output, args.worker_batch)
    elif args.worker is not None:
        worker(args.output, [args.worker])
    elif args.replay is not None:
        source = args.output.resolve()
        target = source/'replays'/f'{args.replay:04d}-{time.time_ns()}'
        target.mkdir(parents=True)
        for name in ('base.yaml', 'search.yaml', 'diagnostic_fixes.py'):
            shutil.copyfile(source/name, target/name)
        if args.patch is not None:
            shutil.copyfile(args.patch, target/'diagnostic_fixes.py')
        if args.profile is not None:
            replay_settings = opt.load_config(target/'search.yaml')
            replay_settings['profile'] = args.profile
            (target/'search.yaml').write_text(yaml.safe_dump(replay_settings, sort_keys=False))
        task = json.loads((source/'candidates'/f'{args.replay:04d}'/'task.json').read_text())
        result = launch(target, args.replay, task['coordinates'], args.timeout)
        print(json.dumps(result, indent=2))
    else:
        main(args)
