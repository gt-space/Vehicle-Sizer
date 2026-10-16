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

from diagnostics.errors import SimulationIncompleteError, failure_details, format_failure

from .parameters import ROOT, SweepCase, _get_candidate, apply_case, load_sweep, make_cases
from reporting.history import flatten_numbers, capture_history, spatial_profiles
from .fields import field_category


def scalar_outputs(result: Any) -> dict:
    """Export all exposed numerical SimResult data and nested design results.

    Only ``history`` and ``initial_state`` are handled by the time-series
    extractor; their numeric fields otherwise create redundant enormous CSVs.
    """
    outputs = {}
    if result is None:
        return outputs
    for name in getattr(result, '__dataclass_fields__', {}):
        if name in {'history', 'initial_state'}:
            continue
        value = getattr(result, name)
        if value is None:
            continue
        if isinstance(value, str):
            outputs[name] = value
        else:
            flatten_numbers(value, name, outputs, arrays=True)
    for name in ('feasible', 'accepted', 'completed', 'apogee_reached'):
        try:
            value = getattr(result, name)
            if value is not None:
                outputs[name] = value
        except (ValueError, TypeError, AttributeError):
            pass
    records = getattr(result, 'constraint_records', {})
    outputs['constraint_summary.violated_count'] = sum(r.margin is not None and r.margin < 0 for r in records.values())
    outputs['constraint_summary.unassessed_count'] = sum(r.margin is None for r in records.values())
    outputs['constraint_summary.assessed_count'] = sum(r.margin is not None for r in records.values())
    design = getattr(result, 'design_summary', None)
    if isinstance(design, dict):
        length = design.get('length')
        diameter = design.get('diameter')
        if isinstance(length, (int, float)) and isinstance(diameter, (int, float)) and diameter > 0:
            outputs['design_summary.fineness_ratio'] = length / diameter
    if getattr(result, 'apogee_reached', False):
        # Simulation ends at apogee, so this is apogee downrange, NOT landing range.
        x = getattr(result, 'final_x', None)
        if isinstance(x, (int, float)):
            outputs['flight.apogee_downrange_m'] = float(x)
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
    if 'state.kinematics.t' in columns:
        columns.remove('state.kinematics.t')
        columns.insert(0, 'state.kinematics.t')
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


def _ignore_wind(spec: dict) -> bool:
    """Validate the sweep-only wind override without changing the vehicle YAML."""
    value = spec.get('sweep', {}).get('ignore_wind', False)
    if type(value) is not bool:
        raise ValueError('sweep.ignore_wind must be true or false')
    return value


def _failed_evaluation(error):
    detail = failure_details(error, phase=getattr(error, 'phase', None))
    warnings = getattr(error, 'warnings', [])
    return dict(status='error', failure=format_failure(error), scalar={},
                history_names=[], history_matrix=None, warnings=warnings,
                errors=[dict(detail, warnings=json.dumps(warnings), traceback=traceback.format_exc())])


def _model_resources(cfg: dict) -> dict:
    """Load read-only lookup models locally, never transfer them between processes."""
    from simulation import project_path
    from FluidTables.PropertyModels import property_sources
    from AeroTables import DragModel

    pure, combustion = property_sources(cfg)
    return {
        'pure_properties': pure,
        'combustion_properties': combustion,
        'aero_model': DragModel(project_path(cfg['aero']['model'])),
    }


def _evaluate(candidate: dict, record_history: bool, compute_loads: bool,
              simulate_fn: Callable | None,
              models: dict | None,
              ignore_wind: bool = False, history_detail: str = 'standard',
              record_profiles: bool = False) -> dict:
    """Run a case and return only serializable scalar/history data, not SimResult."""
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
            simulation_kwargs['record_design_summary'] = True
        result = simulate_fn(candidate, **simulation_kwargs)
        if not result.completed:
            raise SimulationIncompleteError(result)
    except Exception as exc:
        # Keep diagnostics only: partial flight outputs must not enter plots or derivatives.
        return _failed_evaluation(exc)

    scalar = scalar_outputs(result)
    names, matrix = [], None
    if getattr(result, 'history', None):
        try:
            rows, summary = capture_history(result, history_detail)
            scalar.update(summary)
            if record_history:
                names, matrix = _numeric_history(rows)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            issue = f'History export failed: {type(exc).__name__}: {exc}'
            errors.append({'message': issue, 'traceback': traceback.format_exc()})
            scalar['history_error'] = issue
    profiles = {}
    if record_profiles and getattr(result, 'history', None):
        try:
            profiles = spatial_profiles(result.history)
        except (TypeError, ValueError, KeyError) as exc:
            issue = f'Spatial load export failed: {type(exc).__name__}: {exc}'
            errors.append({'message': issue, 'traceback': traceback.format_exc()})
    return {'status': 'completed', 'failure': None, 'scalar': scalar,
            'history_names': names, 'history_matrix': matrix, 'errors': errors,
            'profiles': profiles, 'warnings': getattr(result, 'warnings', [])}


# ProcessPoolExecutor uses spawn (including on macOS). Each process initializes
# its OWN lookup models once; no HDF5 handle or live solver crosses a process
# boundary. The parent alone writes summary CSVs and histories.h5.
_WORKER_STATE = None


def _worker_initializer(base_cfg: dict, parameters: list, record_history: bool,
                        compute_loads: bool, reuse_models: bool,
                        simulate_fn: Callable | None,
                        ignore_wind: bool,
                        history_detail: str = 'standard',
                        record_profiles: bool = False) -> None:
    global _WORKER_STATE
    _WORKER_STATE = dict(base_cfg=base_cfg, parameters=parameters,
                         record_history=record_history, compute_loads=compute_loads,
                         reuse_models=reuse_models, simulate_fn=simulate_fn,
                         models=None,
                         ignore_wind=ignore_wind, history_detail=history_detail,
                         record_profiles=record_profiles)


def _worker_evaluate(case: SweepCase) -> dict:
    state = _WORKER_STATE
    candidate = apply_case(state['base_cfg'], case, state['parameters'])
    # Lazy initialization lets lookup errors be reported per case rather than
    # breaking the entire process pool and losing the rest of the sweep.
    try:
        if state['reuse_models'] and state['simulate_fn'] is None and state['models'] is None:
            state['models'] = _model_resources(state['base_cfg'])
        return _evaluate(candidate, state['record_history'], state['compute_loads'],
                         state['simulate_fn'], state['models'],
                         state['ignore_wind'],
                         state['history_detail'], state['record_profiles'])
    except Exception as exc:
        return _failed_evaluation(exc)


def _case_row(base_cfg: dict, case: SweepCase, parameters: list) -> dict:
    candidate = apply_case(base_cfg, case, parameters)
    row = {'case_id': case.case_id, 'is_baseline': case.is_baseline}
    for p in parameters:
        if p.name in case.overrides:
            row[p.name] = case.overrides[p.name]
        elif p.offset:
            row[p.name] = 0.0
        else:
            try:
                row[p.name] = _get_candidate(candidate, p.path)
            except (KeyError, ValueError):
                row[p.name] = 0.0 if p.path == 'engine.thrust_tilt_deg' else ''
    return row


def _parallel_evaluations(cases: list[SweepCase], base_cfg: dict, parameters: list,
                          record_history: bool, compute_loads: bool,
                          reuse_models: bool, simulate_fn: Callable | None,
                          workers: int,
                          ignore_wind: bool,
                          history_detail: str = 'standard',
                          record_profiles: bool = False):
    """Yield results as jobs finish, with bounded in-flight tasks/memory."""
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx,
        initializer=_worker_initializer,
        initargs=(base_cfg, parameters, record_history, compute_loads,
                  reuse_models, simulate_fn,
                  ignore_wind, history_detail, record_profiles),
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
                    data = _failed_evaluation(exc)
                yield index, case, data
                submit_one()


def run_sweep(base_cfg: dict, spec: dict, *, simulate_fn: Callable | None = None,
              reuse_models: bool = True) -> 'SweepResults':
    """Evaluate baseline + enabled sweep cases; save portable summary and HDF5 histories.

    simulate_fn permits integration testing without running a real flight.
    Existing flight simulations and config objects are never mutated.
    """
    from .results import SweepResults
    parameters, cases = make_cases(base_cfg, spec)
    ignore_wind = _ignore_wind(spec)
    workers = _worker_count(spec, len(cases))
    # A reusable property/aerodynamic source may depend on swept fields.
    # In such cases build resources per candidate so each case actually uses
    # its selected fluid, combustion model, or aerodynamic lookup table.
    resource_paths = ('property_models.', 'aero.model', 'engine.fuel', 'engine.oxidizer')
    if any(path.startswith(resource_paths) for p in parameters
           for path in (p.path, *p.linked_paths)):
        reuse_models = False
    output_cfg = spec.get('outputs', {})
    if not isinstance(output_cfg, dict):
        raise ValueError('outputs must be a mapping')
    record_history = output_cfg.get('record_history', True)
    history_detail = output_cfg.get('history_detail', 'standard')
    if history_detail not in ('standard', 'full'):
        raise ValueError('outputs.history_detail must be standard or full')
    compute_loads = output_cfg.get('compute_loads', False)
    record_profiles = output_cfg.get('record_structural_profiles', False)
    if type(record_profiles) is not bool:
        raise ValueError('outputs.record_structural_profiles must be true or false')
    if record_profiles and not (compute_loads and record_history):
        raise ValueError('record_structural_profiles requires compute_loads and record_history')
    if type(record_history) is not bool or type(compute_loads) is not bool:
        raise ValueError('record_history and compute_loads must be booleans')
    out = _result_directory(spec)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'sweep_config.yaml').write_text(yaml.safe_dump(spec, sort_keys=False))
    (out / 'baseline_config.yaml').write_text(yaml.safe_dump(base_cfg, sort_keys=False))

    print(f'Sensitivity sweep: {len(cases)} cases using {workers} '
          f'{"process" if workers == 1 else "processes"}', flush=True)
    models = None
    summaries = [None] * len(cases)
    cases_table = [None] * len(cases)
    errors, warnings = [], []
    input_columns = [p.name for p in parameters]
    with h5py.File(out / 'histories.h5', 'w') as store:
        store.attrs['format_version'] = 1
        if workers > 1:
            evaluations = _parallel_evaluations(cases, base_cfg, parameters,
                                                record_history, compute_loads,
                                                reuse_models, simulate_fn,
                                                workers,
                                                ignore_wind, history_detail,
                                                record_profiles)
        else:
            def serial_evaluations():
                nonlocal models
                for index, case in enumerate(cases):
                    if reuse_models and simulate_fn is None and models is None:
                        try:
                            models = _model_resources(base_cfg)
                        except Exception as exc:
                            yield index, case, _failed_evaluation(exc)
                            continue
                    candidate = apply_case(base_cfg, case, parameters)
                    yield index, case, _evaluate(candidate, record_history,
                                                 compute_loads, simulate_fn,
                                                 models,
                                                 ignore_wind, history_detail,
                                                record_profiles)
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
            candidate = apply_case(base_cfg, case, parameters)
            input_values = {f'inputs.{p.path}': (_get_candidate(candidate, p.path)
                            if p.offset else row.get(p.name)) for p in parameters}
            input_values.update({f'inputs.{path}': _get_candidate(candidate, path)
                                 for p in parameters for path in p.linked_paths})
            summaries[index] = dict(row, **input_values, status=status,
                                    error=failure or '', **scalar)
            warnings.extend(dict(case_id=case.case_id, **warning) for warning in data.get('warnings', []))
            for error in data['errors']:
                errors.append(dict(case_id=case.case_id, **error))
            matrix = data['history_matrix']
            if record_history and matrix is not None:
                group = store.create_group(case.case_id)
                group.attrs['columns'] = json.dumps(data['history_names'])
                group.create_dataset('data', data=matrix,
                                     compression='gzip' if matrix.shape[0] > 1 else None)
                if data.get('profiles'):
                    profiles_group = group.create_group('profiles')
                    for field_name, profile in data['profiles'].items():
                        profiles_group.create_dataset(field_name, data=profile,
                                                      compression='gzip')
            print(f'[{finished}/{len(cases)}] {case.case_id}: {status} '
                  f'{case.overrides}', flush=True)
            if failure:
                print(failure, flush=True)
    _write_csv(out / 'cases.csv', cases_table, ('case_id', 'is_baseline', *input_columns, 'overrides'))
    _write_csv(out / 'summary.csv', summaries,
               ('case_id', 'is_baseline', 'status', 'error', *input_columns))
    _write_csv(out / 'errors.csv', errors, ('case_id', 'kind', 'exception', 'phase', 'message', 'traceback'))
    _write_csv(out / 'warnings.csv', warnings, ('case_id', 'code', 'message', 'count'))
    catalog = []
    names = sorted({key for row in summaries for key in row})
    for field in names:
        usable = sum(isinstance(row.get(field), (int, float, np.number))
                     and math.isfinite(float(row[field])) for row in summaries)
        catalog.append({'field': field, 'category': field_category(field),
                        'cases_with_numeric_data': usable,
                        'total_cases': len(summaries)})
    _write_csv(out / 'field_catalog.csv', catalog,
               ('field', 'category', 'cases_with_numeric_data', 'total_cases'))
    return SweepResults.load(out)


def run_file(path: str | Path, *, dry_run: bool = False) -> 'SweepResults | list[SweepCase]':
    base_cfg, spec, _ = load_sweep(path)
    if dry_run:
        parameters, cases = make_cases(base_cfg, spec)
        _ignore_wind(spec)
        for case in cases:
            apply_case(base_cfg, case, parameters)
            print(f'{case.case_id}: {case.overrides}')
        print(f'{len(cases)} cases validated (including baseline). No flights run.')
        return cases
    return run_sweep(base_cfg, spec)
