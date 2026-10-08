"""Run independent sensitivity candidates without modifying the normal flight entry point."""
from __future__ import annotations

import csv
import json
import math
import traceback
from dataclasses import dataclass
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


def run_sweep(base_cfg: dict, spec: dict, *, simulate_fn: Callable | None = None,
              history_fn: Callable | None = None, reuse_models: bool = True) -> 'SweepResults':
    """Evaluate baseline + enabled sweep cases; save portable summary and HDF5 histories.

    simulate_fn/history_fn permit integration testing without running a real flight.
    Existing flight simulations and config objects are never mutated.
    """
    from .results import SweepResults
    parameters, cases = make_cases(base_cfg, spec)
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

    models: dict = {}
    if simulate_fn is None:
        from simulation import simulate, property_sources, project_path
        from AeroTables import DragModel
        simulate_fn = simulate
        if reuse_models:
            # Table sources/aero are read-only. Reuse only across fresh vehicle/flight state.
            pure, combustion = property_sources(base_cfg)
            models = dict(pure_properties=pure, combustion_properties=combustion,
                          aero_model=DragModel(project_path(base_cfg['aero']['model'])))
    summaries = []
    cases_table = []
    errors = []
    input_columns = [p.name for p in parameters]
    with h5py.File(out / 'histories.h5', 'w') as store:
        store.attrs['format_version'] = 1
        for index, case in enumerate(cases):
            candidate = apply_case(base_cfg, case, parameters)
            row = {'case_id': case.case_id, 'is_baseline': case.is_baseline}
            # Record all actual independent inputs, not only overrides.
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
                        row[p.name] = (0.0 if p.name == 'thrust_tilt' else '')
            cases_table.append(dict(row, **{'overrides': json.dumps(case.overrides)}))
            result = None
            failure = None
            try:
                result = simulate_fn(candidate, record_history=record_history,
                                     compute_loads=compute_loads, **models)
            except Exception as exc:
                # EvaluationFailure can carry a useful partial SimResult.
                result = getattr(exc, 'partial_result', None)
                failure = f'{type(exc).__name__}: {exc}'
                if exc.__cause__ is not None:
                    failure += f' (caused by {type(exc.__cause__).__name__}: {exc.__cause__})'
                errors.append({'case_id': case.case_id, 'message': failure,
                               'traceback': traceback.format_exc()})
            scalar = scalar_outputs(result) if result is not None else {}
            # A baseline using explicit throat area lacks target_thrust in the
            # config, and vice versa. Give it the actual computed design value.
            if case.is_baseline and result is not None:
                if 'target_thrust' in input_columns and row.get('target_thrust') == '':
                    row['target_thrust'] = scalar.get('design_summary.design_thrust', '')
                if 'throat_area' in input_columns and row.get('throat_area') == '':
                    row['throat_area'] = scalar.get('design_summary.throat_area', '')
                cases_table[-1].update(row)
            term = scalar.get('termination', '')
            status = ('error' if failure else
                      'infeasible' if str(term).startswith('infeasible') or scalar.get('feasible') is False
                      else 'completed')
            summaries.append(dict(row, status=status, error=failure or '', **scalar))
            if record_history and result is not None and result.history:
                try:
                    if history_fn is None:
                        from main import history_rows
                        history_fn = history_rows
                    history = history_fn(result.history)
                    names, matrix = _numeric_history(history)
                    group = store.create_group(case.case_id)
                    group.attrs['columns'] = json.dumps(names)
                    group.create_dataset('data', data=matrix,
                                         compression='gzip' if matrix.shape[0] > 1 else None)
                except Exception as exc:
                    # The flight itself succeeded, but saved history is incomplete.
                    issue = f'History export failed: {type(exc).__name__}: {exc}'
                    errors.append({'case_id': case.case_id, 'message': issue,
                                   'traceback': traceback.format_exc()})
                    summaries[-1]['history_error'] = issue
            print(f'[{index + 1}/{len(cases)}] {case.case_id}: {status} '
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
        for case in cases:
            apply_case(base_cfg, case, parameters)
            print(f'{case.case_id}: {case.overrides}')
        print(f'{len(cases)} cases validated (including baseline). No flights run.')
        return cases
    return run_sweep(base_cfg, spec)
