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

from .parameters import ROOT, SweepCase, _get_candidate, apply_case, load_sweep, make_cases


def _flatten_numeric(value: Any, prefix: str, result: dict, depth: int = 0) -> None:
    """Flatten scalar numeric output; skip heavy flight and non-scalar data."""
    if isinstance(value, (bool, int, float, np.number)):
        try:
            result[prefix] = float(value)
        except (TypeError, ValueError):
            pass
    elif isinstance(value, dict) and depth < 9:
        for key, item in value.items():
            _flatten_numeric(item, f'{prefix}.{key}', result, depth + 1)
    elif isinstance(value, (list, tuple)) and depth < 9:
        for index, item in enumerate(value):
            _flatten_numeric(item, f'{prefix}[{index}]', result, depth + 1)
    elif isinstance(value, np.ndarray) and value.ndim == 0:
        _flatten_numeric(value.item(), prefix, result, depth)



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



def _flat_state_numbers(value: Any, prefix: str, out: dict, *, depth: int = 0) -> None:
    """Export numerical state without serializing solver objects or large arrays."""
    if isinstance(value, (bool, int, float, np.number)):
        if math.isfinite(float(value)):
            out[prefix] = float(value)
    elif isinstance(value, dict) and depth < 6:
        for key, item in value.items():
            _flat_state_numbers(item, f'{prefix}.{key}', out, depth=depth + 1)


def _fluid_nodes(state: dict) -> dict:
    try:
        return state['plant'].fluids.node
    except (KeyError, TypeError, AttributeError):
        return {}


def _fluid_history_rows(history: list, detail: str = 'standard') -> list[dict]:
    """Expose every node's thermodynamic unknowns plus optional full reporting data.

    Fields are based on actual returned solver states, so different propulsion
    templates naturally export different nodes without special-case dispatch.
    """
    rows = []
    for state in history:
        row = {}
        try:
            fluids = state['plant'].fluids
        except (KeyError, AttributeError, TypeError):
            rows.append(row)
            continue
        for node_id, node in fluids.node.items():
            for field, value in node.items():
                if detail == 'full':
                    _flat_state_numbers(value, f'fluid.node.{node_id}.{field}', row)
                elif field in ('P', 'T', 'T_liq', 'T_ull', 'm', 'm_liq', 'm_ull',
                               'U', 'U_liq', 'U_ull', 'quality'):
                    _flat_state_numbers(value, f'fluid.node.{node_id}.{field}', row)
        for branch_id, mdot in fluids.mdot.items():
            _flat_state_numbers(mdot, f'fluid.branch.{branch_id}.mdot', row)
        if detail == 'full':
            for branch_id, branch in fluids.branch.items():
                _flat_state_numbers(branch, f'fluid.branch.{branch_id}', row)
            thermal = state['plant'].thermal
            if thermal is not None:
                _flat_state_numbers(thermal.node, 'thermal.node', row)
        rows.append(row)
    return rows


def _fluid_statistics(history: list, result: Any) -> dict:
    """Derive actual initial, minimum, maximum and powered-EOL node states.

    EOL is the first accepted flight endpoint at engine shutdown, not apogee.
    When no shutdown is recorded EOL is unavailable, not fabricated.
    """
    if not history or not all(isinstance(state, dict) and 'kinematics' in state for state in history):
        return {}
    eol_index = None
    for i, state in enumerate(history):
        if i and history[i-1].get('engine_on') and not state.get('engine_on'):
            eol_index = i
            break
    if eol_index is None and getattr(result, 'burn_complete', False) and getattr(result, 'burn_duration', 0) > 0:
        duration = getattr(result, 'burn_duration', None)
        if duration is not None:
            eol_index = min(range(len(history)), key=lambda i: abs(history[i]['kinematics'].t - duration))
    nodes = [_fluid_nodes(state) for state in history]
    initial = _fluid_nodes(getattr(result, 'initial_state', None))
    names = sorted({name for node in [initial, *nodes] for name in node})
    summary = {}
    if eol_index is not None:
        summary['fluid.eol_time_s'] = float(history[eol_index]['kinematics'].t)
    for name in names:
        measurements = {}
        for i, node_map in [(-1, initial), *enumerate(nodes)]:
            node = node_map.get(name, {})
            if not isinstance(node, dict):
                continue
            # Volume-component masses/energies and P/T are physical state variables.
            # Use NodeState.as_dict() results; never infer an EOS from P alone.
            for field in ('P', 'T', 'T_ull', 'T_liq', 'm', 'm_ull', 'm_liq',
                          'U', 'U_ull', 'U_liq', 'quality'):
                value = node.get(field)
                if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
                    measurements.setdefault(field, []).append((i, float(value)))
        for field, values in measurements.items():
            by_index = dict(values)
            prefix = f'fluid.{name}'
            summary[f'{prefix}.initial.{field}'] = values[0][1]
            summary[f'{prefix}.final.{field}'] = values[-1][1]
            summary[f'{prefix}.minimum.{field}'] = min(v for _, v in values)
            summary[f'{prefix}.maximum.{field}'] = max(v for _, v in values)
            if eol_index in by_index:
                summary[f'{prefix}.eol.{field}'] = by_index[eol_index]
        for nickname, field in [('pressure', 'P'), ('temperature', 'T'),
                                ('gas_temperature', 'T_ull'), ('gas_mass', 'm_ull'),
                                ('mass', 'm')]:
            key = f'fluid.{name}.eol.{field}'
            if key in summary:
                summary[f'fluid.{name}.eol_{nickname}'] = summary[key]
    return summary

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
              ignore_wind: bool = False, history_detail: str = 'standard') -> dict:
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
            simulation_kwargs['record_design_summary'] = True
        result = simulate_fn(candidate, **simulation_kwargs)
    except Exception as exc:
        result = getattr(exc, 'partial_result', None)
        failure = f'{type(exc).__name__}: {exc}'
        if exc.__cause__ is not None:
            failure += f' (caused by {type(exc.__cause__).__name__}: {exc.__cause__})'
        errors.append({'message': failure, 'traceback': traceback.format_exc()})

    scalar = scalar_outputs(result) if result is not None else {}
    if result is not None and getattr(result, 'history', None):
        try:
            scalar.update(_fluid_statistics(result.history, result))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            issue = f'Fluid summary export failed: {type(exc).__name__}: {exc}'
            errors.append({'message': issue, 'traceback': traceback.format_exc()})
            scalar['fluid_summary_error'] = issue
    status = _case_status(scalar, failure, ignore_feasibility)
    names, matrix = [], None
    if record_history and result is not None and getattr(result, 'history', None):
        try:
            if history_fn is None:
                from main import history_rows as history_fn
            rows = history_fn(result.history)
            extra = _fluid_history_rows(result.history, history_detail)
            if len(rows) != len(extra):
                raise ValueError('Flight history and fluid-state history lengths differ')
            names, matrix = _numeric_history([{**row, **fluid} for row, fluid in zip(rows, extra)])
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
                        ignore_feasibility: bool, ignore_wind: bool,
                        history_detail: str = 'standard') -> None:
    global _WORKER_STATE
    _WORKER_STATE = dict(base_cfg=base_cfg, parameters=parameters,
                         record_history=record_history, compute_loads=compute_loads,
                         reuse_models=reuse_models, simulate_fn=simulate_fn,
                         history_fn=history_fn, models=None,
                         ignore_feasibility=ignore_feasibility,
                         ignore_wind=ignore_wind, history_detail=history_detail)


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
                         state['ignore_feasibility'], state['ignore_wind'],
                         state['history_detail'])
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
                row[p.name] = _get_candidate(candidate, p.path)
            except (KeyError, ValueError):
                row[p.name] = 0.0 if p.path == 'engine.thrust_tilt_deg' else ''
    return row


def _parallel_evaluations(cases: list[SweepCase], base_cfg: dict, parameters: list,
                          record_history: bool, compute_loads: bool,
                          reuse_models: bool, simulate_fn: Callable | None,
                          history_fn: Callable | None, workers: int,
                          ignore_feasibility: bool, ignore_wind: bool,
                          history_detail: str = 'standard'):
    """Yield results as jobs finish, with bounded in-flight tasks/memory."""
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx,
        initializer=_worker_initializer,
        initargs=(base_cfg, parameters, record_history, compute_loads,
                  reuse_models, simulate_fn, history_fn, ignore_feasibility,
                  ignore_wind, history_detail),
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
    # A reusable property/aerodynamic source may depend on swept fields.
    # In such cases build resources per candidate so each case actually uses
    # its selected fluid, combustion model, or aerodynamic lookup table.
    resource_paths = ('property_models.', 'aero.model', 'engine.fuel', 'engine.oxidizer')
    if any(p.path.startswith(resource_paths) for p in parameters):
        reuse_models = False
    output_cfg = spec.get('outputs', {})
    if not isinstance(output_cfg, dict):
        raise ValueError('outputs must be a mapping')
    record_history = output_cfg.get('record_history', True)
    history_detail = output_cfg.get('history_detail', 'standard')
    if history_detail not in ('standard', 'full'):
        raise ValueError('outputs.history_detail must be standard or full')
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
                                                ignore_feasibility, ignore_wind, history_detail)
        else:
            def serial_evaluations():
                for index, case in enumerate(cases):
                    candidate = apply_case(base_cfg, case, parameters)
                    yield index, case, _evaluate(candidate, record_history,
                                                 compute_loads, simulate_fn,
                                                 history_fn, models,
                                                 ignore_feasibility, ignore_wind, history_detail)
            evaluations = serial_evaluations()

        for finished, (index, case, data) in enumerate(evaluations, start=1):
            row = _case_row(base_cfg, case, parameters)
            scalar = data['scalar']
            failure = data['failure']
            status = data['status']
            # A baseline using explicit throat area lacks target_thrust in the
            # config, and vice versa. Give it the actual computed design value.
            if case.is_baseline and scalar:
                for p in parameters:
                    if row.get(p.name) != '':
                        continue
                    if p.path == 'prop_system.thrust_target':
                        row[p.name] = scalar.get('design_summary.design_thrust', '')
                    elif p.path == 'engine.throat_area':
                        row[p.name] = scalar.get('design_summary.throat_area', '')
            cases_table[index] = dict(row, overrides=json.dumps(case.overrides))
            input_values = {f'inputs.{p.path}': row.get(p.name) for p in parameters}
            summaries[index] = dict(row, **input_values, status=status,
                                    error=failure or '', **scalar)
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
