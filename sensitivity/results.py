"""Reload and query saved sweep summaries and flight histories independently of physics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from .derivatives import gradient, history_gradient


def _parse_cell(value: str):
    if value in ('', None):
        return None
    if value in ('True', 'true'):
        return True
    if value in ('False', 'false'):
        return False
    try:
        return float(value)
    except ValueError:
        return value


class SweepResults:
    def __init__(self, folder: Path, summary: list[dict], spec: dict):
        self.folder = folder
        self.summary = summary
        self.spec = spec
        self.mode = spec['sweep']['mode']
        self.parameters = [name for name, settings in spec['sweep']['parameters'].items()
                           if settings.get('enabled', False)]
        self.by_id = {row['case_id']: row for row in summary}
        # Track actual independent OAT changes, not incidental changes in
        # derived design values reported for virtual sizing inputs.
        with (folder / 'cases.csv').open(newline='') as stream:
            self.overrides = {record['case_id']: json.loads(record['overrides'])
                              for record in csv.DictReader(stream)}

    @classmethod
    def load(cls, folder: str | Path) -> 'SweepResults':
        folder = Path(folder)
        with (folder / 'sweep_config.yaml').open() as stream:
            spec = yaml.safe_load(stream)
        with (folder / 'summary.csv').open(newline='') as stream:
            summary = [{key: _parse_cell(value) for key, value in row.items()}
                       for row in csv.DictReader(stream)]
        return cls(folder, summary, spec)

    def fields(self) -> list[str]:
        return sorted({key for row in self.summary for key in row})

    def history_fields(self) -> list[str]:
        with h5py.File(self.folder / 'histories.h5') as store:
            return sorted({key for group in store.values()
                           for key in json.loads(group.attrs['columns'])})

    def history(self, case_id: str) -> dict[str, np.ndarray]:
        with h5py.File(self.folder / 'histories.h5') as store:
            if case_id not in store:
                raise KeyError(f'No recorded flight history for {case_id}')
            group = store[case_id]
            columns = json.loads(group.attrs['columns'])
            values = group['data'][:]
        return {name: values[:, index] for index, name in enumerate(columns)}

    def _select(self, *, varying: str | None = None, filters: dict | None = None,
                include_baseline: bool = True, include_infeasible: bool = False) -> list[dict]:
        filters = filters or {}
        unknown = set(filters) - set(self.parameters)
        if unknown:
            raise ValueError(f'Filter unknown sweep parameters: {sorted(unknown)}')
        if varying in self.parameters:
            if self.mode == 'full_grid':
                missing = set(self.parameters) - {varying} - set(filters)
                if missing:
                    raise ValueError(f'Full-grid plot varying {varying!r} needs filters for {sorted(missing)}')
            elif filters:
                # OAT changes only one parameter, so filtering to a distinct baseline
                # value is generally unnecessary but can select a subset.
                pass
        selected = []
        for row in self.summary:
            if row['status'] == 'error' or (row['status'] == 'infeasible' and not include_infeasible):
                continue
            if row['is_baseline']:
                if not include_baseline:
                    continue
                if any(row.get(name) != value for name, value in filters.items()):
                    continue
            else:
                if self.mode == 'one_at_a_time' and varying in self.parameters:
                    # At most one sweep param differs from baseline in OAT.
                    if set(self.overrides.get(row['case_id'], {})) - {varying}:
                        continue
                if any(row.get(name) != value for name, value in filters.items()):
                    continue
            selected.append(row)
        return selected

    def scalar_points(self, x: str, y: str, *, filters: dict | None = None,
                      include_infeasible: bool = False):
        varying = x if x in self.parameters else None
        cases = self._select(varying=varying, filters=filters,
                             include_baseline=(self.mode == 'one_at_a_time' or varying is None),
                             include_infeasible=include_infeasible)
        points = []
        for row in cases:
            xx, yy = row.get(x), row.get(y)
            if isinstance(xx, (float, int)) and isinstance(yy, (float, int)):
                if np.isfinite(xx) and np.isfinite(yy):
                    points.append((float(xx), float(yy), row['case_id']))
        return sorted(points)

    def scalar_derivative(self, y: str, wrt: str, *, filters: dict | None = None,
                          include_infeasible: bool = False):
        if wrt not in self.parameters:
            raise ValueError(f'{wrt!r} is not an enabled sweep input')
        cases = self._select(varying=wrt, filters=filters,
                             include_baseline=self.mode == 'one_at_a_time',
                             include_infeasible=include_infeasible)
        pairs = sorted((float(row[wrt]), float(row[y])) for row in cases
                       if isinstance(row.get(wrt), (int, float)) and isinstance(row.get(y), (int, float))
                       and np.isfinite(row[wrt]) and np.isfinite(row[y]))
        unique = {}
        for x, value in pairs:
            if x in unique and not np.isclose(unique[x], value, rtol=1e-7, atol=1e-10):
                raise ValueError('Different outputs at the same input value; filter other parameters')
            unique[x] = value
        x = np.array(sorted(unique))
        return x, gradient(x, [unique[xx] for xx in x])

    def history_derivative(self, y: str, wrt: str, *, filters: dict | None = None,
                           include_infeasible: bool = False):
        if wrt not in self.parameters:
            raise ValueError(f'{wrt!r} is not an enabled sweep input')
        cases = self._select(varying=wrt, filters=filters,
                             include_baseline=self.mode == 'one_at_a_time',
                             include_infeasible=include_infeasible)
        samples = []
        seen = set()
        for case in cases:
            x = case.get(wrt)
            if not isinstance(x, (int, float)) or x in seen:
                continue
            try:
                history = self.history(case['case_id'])
            except KeyError:
                continue
            seen.add(x)
            samples.append((x, history))
        return history_gradient(samples, y)

    def plot(self, x: str, y: str, **kwargs):
        from .plotting import plot_scalar
        return plot_scalar(self, x, y, **kwargs)

    def plot_history(self, y: str, group_by: str, **kwargs):
        from .plotting import plot_history
        return plot_history(self, y, group_by, **kwargs)

    def plot_derivative(self, x: str, y: str, wrt: str, **kwargs):
        from .plotting import plot_derivative
        return plot_derivative(self, x, y, wrt, **kwargs)

    def plot_history_derivative(self, y: str, wrt: str, **kwargs):
        from .plotting import plot_history_derivative
        return plot_history_derivative(self, y, wrt, **kwargs)
