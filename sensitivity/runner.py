"""Run independent sensitivity candidates without modifying the normal flight entry point."""
from __future__ import annotations

import csv
import json
import math
import multiprocessing
import os
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import yaml

from .parameters import ROOT, SweepCase, apply_case, load_sweep, make_cases


def _flatten_numeric(value: Any, prefix: str, result: dict, depth: int = 0) -> None:
    """Flatten scalar numeric output; skip heavy flight and non-scalar data."""
    if isinstance(value, (bool, int, float, np.number)):
        try:
            result[prefix] = float(value)
        except (TypeError, ValueError):
            pass
    elif isinstance(value, dict) and depth < 6:
        for key, item in value.items():
            _flatten_numeric(item, f'{prefix}.{key}', result, depth + 1)
    # Lists of tank section structures in design_summary aren't flat scalar outputs.


def scalar_outputs(result: Any) -> dict:
    """Extract named scalar fields and nested numeric diagnostics from SimResult."""
    outputs = {}
    for name in getattr(result, '__dataclass_fields__', {}):
        if name in {'history', 'initial_state', 'warnings', 'constraint_records'}:
            continue
        value = getattr(result, name)
        if value is None:
            continue
        if isinstance(value, (str, bool)):
            outputs[name] = value
        elif isinstance(value, (int, float, np.number)):
            outputs[name] = float(value)
        elif isinstance(value, dict):
            _flatten_numeric(value, name, outputs)
    for name in ('feasible', 'accepted', 'completed', 'apogee_reached'):
        try:
            value = getattr(result, name)
            if value is not None:
                outputs[name] = value
        except (ValueError, TypeError):
            pass
    return outputs


def _write_csv(path: Path, rows: list[dict], leading: tuple[str, ...]) -> None:
    fields = list(dict.fromkeys([*leading, *(k for row in rows for k in row)]))
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _numeric_history(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    if not rows:
        return [], np.empty((0, 0))
    columns = sorted({key for row in rows for key, value in row.items()
                      if isinstance(value, (int, float, bool, np.number))})
    if 'time_s' in columns:
        columns.remove('time_s')
        columns.insert(0, 'time_s')
    matrix = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, field in enumerate(columns):
            value = row.get(field)
            if isinstance(value, (bool, int, float, np.number)):
                matrix[i, j] = float(value)
    return columns, matrix


def _result_directory(spec: dict) -> Path:
    entry = spec.get('outputs', {}).get('directory', 'outputs/sensitivity')
    if not isinstance(entry, str) or not entry:
        raise ValueError('outputs.directory must be a nonempty path')
    path = Path(entry).expanduser()
    return path if path.is_absolute() else ROOT / path


def _worker_count(spec: dict, case_count: int) -> int:
    """Validate the number of processes (default 1 for backward compatibility).

    'auto' uses at most four processes because each process holds independent,
    potentially large fluid and aerodynamic lookup tables in memory.
    """
    requested = spec.get('sweep', {}).get('workers', 1)
    if requested == 'auto':
        requested = max(1, min(4, (os.cpu_count() or 2) - 1))
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError('sweep.workers must be a positive integer or auto')
    return min(requested, case_count)


def _sweep_flag(spec: dict, key: str) -> bool:
    """Validate an opt-in sweep-only boolean without changing the vehicle YAML."""
    value = spec.get('sweep', {}).get(key, False)
    if type(value) is not bool:
        raise ValueError(f'sweep.{key} must be true or false')
    return value


def _ignore_feasibility(spec: dict) -> bool:
    return _sweep_flag(spec, 'ignore_feasibility')


def _ignore_wind(spec: dict) -> bool:
    return _sweep_flag(spec, 'ignore_wind')


def _case_status(scalar: dict, failure: str | None, ignore_feasibility: bool) -> str:
    """Classify flight execution separately from engineering feasibility.

    Do not mask a solver error, rejected initial design, or premature operating
    termination. Only a flight that reached apogee may bypass feasibility in a
    sensitivity sweep. The actual SimResult.feasible and all constraint margins
    are preserved unchanged in the saved scalar outputs.
    """
    if failure:
        return 'error'
    term = str(scalar.get('termination', ''))
    if term.startswith('infeasible'):
        return 'infeasible'
    if ignore_feasibility:
        return 'completed' if scalar.get('completed') is True else 'infeasible'
    return 'infeasible' if scalar.get('feasible') is False else 'completed'


def _model_resources(cfg: dict) -> dict:
    """Load read-only lookup models locally, never transfer them between processes."""
    from simulation import property_sources, project_path
    from AeroTables import DragModel

    pure, combustion = property_sources(cfg)
    return {
        'pure_properties': pure,
        'combustion_properties': combustion,
        'aero_model': DragModel(project_path(cfg['aero']['model'])),
    }


def _evaluate(candidate: dict, record_history: bool, compute_loads: bool,
              simulate_fn: Callable | None, history_fn: Callable | None,
              models: dict | None, ignore_feasibility: bool = False,
              ignore_wind: bool = False) -> dict:
    """Run a case and return only serializable scalar/history data, not SimResult."""
    result = None
    failure = None
    errors = []
    try:
        if ignore_wind:
            # apply_case already created a fresh candidate: never mutate base_cfg.
            environment = candidate.setdefault('environment', {})
            wind = environment.setdefault('wind', {})
            if not isinstance(wind, dict):
                raise ValueError('environment.wind must be a mapping')
            wind['enabled'] = False
        simulation_kwargs = dict(record_history=record_history,
                                 compute_loads=compute_loads, **(models or {}))
        if simulate_fn is None:
            from simulation import simulate as simulate_fn
            simulation_kwargs['ignore_feasibility'] = ignore_feasibility
        result = simulate_fn(candidate, **simulation_kwargs)
    except Exception as exc:
        result = getattr(exc, 'partial_result', None)
        failure = f'{type(exc).__name__}: {exc}'
        if exc.__cause__ is not None:
            failure += f' (caused by {type(exc.__cause__).__name__}: {exc.__cause__})'
        errors.append({'message': failure, 'traceback': traceback.format_exc()})

    scalar = scalar_outputs(result) if result is not None else {}
    status = _case_status(scalar, failure, ignore_feasibility)
    names, matrix = [], None
    if record_history and result is not None and getattr(result, 'history', None):
        try:
            if history_fn is None:
                from main import history_rows as history_fn
            names, matrix = _numeric_history(history_fn(result.history))
        except Exception as exc:
            issue = f'History export failed: {type(exc).__name__}: {exc}'
            errors.append({'message': issue, 'traceback': traceback.format_exc()})
            scalar['history_error'] = issue
    return {'status': status, 'failure': failure, 'scalar': scalar,
            'history_names': names, 'history_matrix': matrix, 'errors': errors}


# ProcessPoolExecutor uses spawn (including on macOS). Each process initializes
# its OWN lookup models once; no HDF5 handle or live solver crosses a process
# boundary. The parent alone writes summary CSVs and histories.h5.
_WORKER_STATE = None


def _worker_initializer(base_cfg: dict, parameters: list, record_history: bool,
                        compute_loads: bool, reuse_models: bool,
                        simulate_fn: Callable | None, history_fn: Callable | None,
                        ignore_feasibility: bool, ignore_wind: bool) -> None:
    global _WORKER_STATE
    _WORKER_STATE = dict(base_cfg=base_cfg, parameters=parameters,
                         record_history=record_history, compute_loads=compute_loads,
                         reuse_models=reuse_models, simulate_fn=simulate_fn,
                         history_fn=history_fn, models=None,
                         ignore_feasibility=ignore_feasibility,
                         ignore_wind=ignore_wind)


def _worker_evaluate(case: SweepCase) -> dict:
    state = _WORKER_STATE
    candidate = apply_case(state['base_cfg'], case, state['parameters'])
    # Lazy initialization lets lookup errors be reported per case rather than
    # breaking the entire process pool and losing the rest of the sweep.
    try:
        if state['reuse_models'] and state['simulate_fn'] is None and state['models'] is None:
            state['models'] = _model_resources(state['base_cfg'])
        return _evaluate(candidate, state['record_history'], state['compute_loads'],
                         state['simulate_fn'], state['history_fn'], state['models'],
                         state['ignore_feasibility'], state['ignore_wind'])
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'
        return {'status': 'error', 'failure': message, 'scalar': {},
                'history_names': [], 'history_matrix': None,
                'errors': [{'message': message, 'traceback': traceback.format_exc()}]}


def _case_row(base_cfg: dict, case: SweepCase, parameters: list) -> dict:
    candidate = apply_case(base_cfg, case, parameters)
    row = {'case_id': case.case_id, 'is_baseline': case.is_baseline}
    for p in parameters:
        if p.name in case.overrides:
            row[p.name] = case.overrides[p.name]
        else:
            try:
                node = candidate
                for part in p.path.split('.'):
                    node = node[part]
                row[p.name] = node
            except KeyError:
                row[p.name] = 0.0 if p.name == 'thrust_tilt' else ''
    return row


def _parallel_evaluations(cases: list[SweepCase], base_cfg: dict, parameters: list,
                          record_history: bool, compute_loads: bool,
                          reuse_models: bool, simulate_fn: Callable | None,
                          history_fn: Callable | None, workers: int,
                          ignore_feasibility: bool, ignore_wind: bool):
    """Yield results as jobs finish, with bounded in-flight tasks/memory."""
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx,
        initializer=_worker_initializer,
        initargs=(base_cfg, parameters, record_history, compute_loads,
                  reuse_models, simulate_fn, history_fn, ignore_feasibility, ignore_wind),
    ) as pool:
        todo = iter(enumerate(cases))
        pending = {}

        def submit_one():
            try:
                index, case = next(todo)
            except StopIteration:
                return False
            pending[pool.submit(_worker_evaluate, case)] = (index, case)
            return True

        # Keep at most 2*workers payloads in flight, rather than shipping every
        # candidate and potentially buffering gigabytes of flight histories.
        for _ in range(min(len(cases), 2 * workers)):
            submit_one()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                index, case = pending.pop(future)
                try:
                    data = future.result()
                except Exception as exc:
                    message = f'Worker failed: {type(exc).__name__}: {exc}'
                    data = {'status': 'error', 'failure': message, 'scalar': {},
                            'history_names': [], 'history_matrix': None,
                            'errors': [{'message': message, 'traceback': traceback.format_exc()}]}
                yield index, case, data
                submit_one()


def run_sweep(base_cfg: dict, spec: dict, *, simulate_fn: Callable | None = None,
              history_fn: Callable | None = None, reuse_models: bool = True) -> 'SweepResults':
    """Evaluate baseline + enabled sweep cases; save portable summary and HDF5 histories.

    simulate_fn/history_fn permit integration testing without running a real flight.
    Existing flight simulations and config objects are never mutated.
    """
    from .results import SweepResults
    parameters, cases = make_cases(base_cfg, spec)
    ignore_feasibility = _ignore_feasibility(spec)
    ignore_wind = _ignore_wind(spec)
    workers = _worker_count(spec, len(cases))
    output_cfg = spec.get('outputs', {})
    if not isinstance(output_cfg, dict):
        raise ValueError('outputs must be a mapping')
    record_history = output_cfg.get('record_history', True)
    compute_loads = output_cfg.get('compute_loads', False)
    if type(record_history) is not bool or type(compute_loads) is not bool:
        raise ValueError('record_history and compute_loads must be booleans')
    out = _result_directory(spec)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sweep_config.yaml').write_text(yaml.safe_dump(spec, sort_keys=False))
    (out / 'baseline_config.yaml').write_text(yaml.safe_dump(base_cfg, sort_keys=False))

    print(f'Sensitivity sweep: {len(cases)} cases using {workers} '
          f'{"process" if workers == 1 else "processes"}', flush=True)
    models = _model_resources(base_cfg) if workers == 1 and reuse_models and simulate_fn is None else None
    summaries = [None] * len(cases)
    cases_table = [None] * len(cases)
    errors = []
    input_columns = [p.name for p in parameters]
    with h5py.File(out / 'histories.h5', 'w') as store:
        store.attrs['format_version'] = 1
        if workers > 1:
            evaluations = _parallel_evaluations(cases, base_cfg, parameters,
                                                record_history, compute_loads,
                                                reuse_models, simulate_fn,
                                                history_fn, workers,
                                                ignore_feasibility, ignore_wind)
        else:
            def serial_evaluations():
                for index, case in enumerate(cases):
                    candidate = apply_case(base_cfg, case, parameters)
                    yield index, case, _evaluate(candidate, record_history,
                                                 compute_loads, simulate_fn,
                                                 history_fn, models,
                                                 ignore_feasibility, ignore_wind)
            evaluations = serial_evaluations()

        for finished, (index, case, data) in enumerate(evaluations, start=1):
            row = _case_row(base_cfg, case, parameters)
            scalar = data['scalar']
            failure = data['failure']
            status = data['status']
            # A baseline using explicit throat area lacks target_thrust in the
            # config, and vice versa. Give it the actual computed design value.
            if case.is_baseline and scalar:
                if 'target_thrust' in input_columns and row.get('target_thrust') == '':
                    row['target_thrust'] = scalar.get('design_summary.design_thrust', '')
                if 'throat_area' in input_columns and row.get('throat_area') == '':
                    row['throat_area'] = scalar.get('design_summary.throat_area', '')
            cases_table[index] = dict(row, overrides=json.dumps(case.overrides))
            summaries[index] = dict(row, status=status, error=failure or '', **scalar)
            for error in data['errors']:
                errors.append(dict(case_id=case.case_id, **error))
            matrix = data['history_matrix']
            if record_history and matrix is not None:
                group = store.create_group(case.case_id)
                group.attrs['columns'] = json.dumps(data['history_names'])
                group.create_dataset('data', data=matrix,
                                     compression='gzip' if matrix.shape[0] > 1 else None)
            print(f'[{finished}/{len(cases)}] {case.case_id}: {status} '
                  f'{case.overrides}', flush=True)
    _write_csv(out / 'cases.csv', cases_table, ('case_id', 'is_baseline', *input_columns, 'overrides'))
    _write_csv(out / 'summary.csv', summaries,
               ('case_id', 'is_baseline', 'status', 'error', *input_columns))
    _write_csv(out / 'errors.csv', errors, ('case_id', 'message', 'traceback'))
    return SweepResults.load(out)


def run_file(path: str | Path, *, dry_run: bool = False) -> 'SweepResults | list[SweepCase]':
    base_cfg, spec, _ = load_sweep(path)
    if dry_run:
        parameters, cases = make_cases(base_cfg, spec)
        _ignore_feasibility(spec)  # Validate both sweep options before running.
        _ignore_wind(spec)
        for case in cases:
            apply_case(base_cfg, case, parameters)
            print(f'{case.case_id}: {case.overrides}')
        print(f'{len(cases)} cases validated (including baseline). No flights run.')
        return cases
    return run_sweep(base_cfg, spec)
