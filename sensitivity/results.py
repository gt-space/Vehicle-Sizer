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
                           if settings.get('enabled', True)]
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

    def fields(self, contains: str | None = None) -> list[str]:
        """List saved scalar field names; optionally filter by substring."""
        names = sorted({key for row in self.summary for key in row})
        return names if contains is None else [name for name in names if contains.lower() in name.lower()]

    def history_fields(self, contains: str | None = None) -> list[str]:
        """List history columns; optionally filter by substring."""
        with h5py.File(self.folder / 'histories.h5') as store:
            names = sorted({key for group in store.values()
                            for key in json.loads(group.attrs['columns'])})
        return names if contains is None else [name for name in names if contains.lower() in name.lower()]

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

    def plot(self, x: str, y=None, **kwargs):
        """Plot one or more scalar fields, optionally using two y-axes.

        ``x`` is a field from fields(). ``y``/``y2`` may each be a field name,
        a list of field names, or {field_name: custom_legend_label}.
        ``y2`` creates a right-side y-axis. ``y2label``, ``y2scale``, ``y2lim``
        configure it. ``labels`` is an optional mapping overriding legend names.
        Single-trace plots default to black; multi-trace plots use colors.

        Existing single-trace calls and saving behavior are unchanged.

        Keyword options (all optional):
            kind: 'auto' (line for swept x, scatter otherwise), 'line', 'scatter'.
            title, xlabel, ylabel: Custom plot title and axis labels.
            figsize: Figure size (width, height) in inches; default (8, 5).
            grid: Show grid; default True.
            legend, legend_title: Legend visibility/heading.
            xlim, ylim: Optional (min, max) axis bounds.
            xscale, yscale: Positive numeric display divisors (e.g. 1e6, 1e3),
                            or Matplotlib scale names ('linear', 'log', etc.).
            filters: Dict fixing other enabled sweep parameters in full_grid.
            include_infeasible: Include infeasible rows with numeric outputs.
            save: False/None to skip saving; True saves under <results_dir>/plots/;
                  str/Path saves to that exact path (relative to working directory).
            filename: Custom filename saved in <results_dir>/plots/ (or absolute
                      path). Providing filename implies saving unless save=False.
            file_format: File type such as 'png', 'jpeg', 'jpg', 'svg', or 'pdf'.
                         Overrides any filename suffix, if given.
            Single-curve lines/scatters default to black; color= overrides it.
            show: Show Matplotlib window; None shows unsaved new figures.
            dpi: Image DPI (default 160); ax: existing matplotlib Axes.
            **style: Matplotlib styling, e.g. color, marker, linewidth,
                     linestyle, alpha, label (as supported by plot/scatter).

        Returns:
            matplotlib.figure.Figure: Figure for further customization.
        """
        from .plotting import plot_scalar
        return plot_scalar(self, x, y, **kwargs)

    def plot_case_history(self, y=None, *, x='time_s', y2=None, case_id='baseline', **kwargs):
        """Plot one saved flight's history using multiple lines and/or two y-axes.

        Example: plot_case_history(y='thrust_N', y2='altitude_m',
        y2scale=1e3, ylabel='Thrust (N)', y2label='Altitude (km)').
        y/y2 can be strings, lists, or {field: legend_label} dictionaries.
        Existing plot_history(y, group_by) is for comparing a sweep of cases.
        """
        from .plotting import plot_case_history
        return plot_case_history(self, y, x=x, y2=y2, case_id=case_id, **kwargs)

    def plot_history(self, y: str, group_by: str, **kwargs):
        """Plot flight-history ``y`` against time, grouped by sweep parameter.

        ``y`` must name a numeric column in ``history_fields()``. ``group_by``
        must be an enabled sweep parameter. Additional keyword options:
            title, xlabel, ylabel: Custom title and axis labels.
            figsize: Figure size (width, height), default (8, 5).
            grid, legend, legend_title: Grid/legend formatting.
            xlim, ylim: Bounds in displayed units. xscale, yscale: Positive
                        divisors for plotted values or 'linear'/'log'/etc.
            filters: Dict holding other full_grid sweep inputs fixed.
            include_infeasible: Include infeasible cases with histories.
            case_labels: Dict mapping case IDs or sweep values to labels.
            colorbar: None (automatic for >8 curves), True, or False.
                      Large sweeps use a numeric colormap instead of a huge legend;
                      the dashed black line identifies the baseline.
            cmap: Matplotlib colormap name; default 'viridis'.
            color_scale: Divide sweep values for colorbar display, e.g. 1e6
                         to display chamber pressure in MPa rather than Pa.
            colorbar_label: Custom colorbar title matching color_scale.
            active_only: Automatically zoom the time axis around nonzero data;
                         helpful for thrust histories with a long zero tail.
            max_traces: Optional upper bound on number of drawn cases; an even
                        subset of numeric sweep values is chosen.
            save: None/False (do not save), True (auto PNG), or filename/path.
            filename, file_format: Custom name and saved image format; see plot().
            show: True/False for interactive display; None is automatic.
            dpi: Saved-image DPI (default 160); ax: existing matplotlib Axes.
            **style: Matplotlib line styling such as color, linewidth, alpha.

        Returns:
            matplotlib.figure.Figure: Figure for further customization.
        """
        from .plotting import plot_history
        return plot_history(self, y, group_by, **kwargs)

    def plot_derivative(self, x: str, y: str, wrt: str, **kwargs):
        """Plot numerical d(``y``)/d(``wrt``) against numeric scalar ``x``.

        ``wrt`` must be an enabled continuous sweep input. Inputs/output field
        names are discoverable with ``fields()``. Additional keyword options:
            kind: 'line' (default) or 'scatter'.
            title, xlabel, ylabel: Custom title and axis labels.
            figsize: Figure size in inches, default (8, 5).
            grid, legend, legend_title: Grid/legend formatting.
            xlim, ylim: Bounds in displayed units. xscale, yscale: Positive
                        divisors for plotted values or 'linear'/'log'/etc.
            filters: Hold other full_grid sweep parameters fixed.
            include_infeasible: Include infeasible cases with valid data.
            save: None/False (no file), True (auto PNG), or filename/path.
            filename, file_format: Custom name and saved image format; see plot().
            Numeric xscale and yscale transform displayed values, but do not
            change the derivative's underlying units (e.g. d(m)/d(Pa)).
            The single derivative curve defaults to black (color overrides).
            show: True/False for interactive display; None is automatic.
            dpi: Image DPI, default 160; ax: existing matplotlib Axes.
            **style: Matplotlib styling such as color, marker, linewidth.

        Returns:
            matplotlib.figure.Figure: Figure for further customization.
        """
        from .plotting import plot_derivative
        return plot_derivative(self, x, y, wrt, **kwargs)

    def plot_history_derivative(self, y: str, wrt: str, **kwargs):
        """Plot d(``y``(t))/d(``wrt``) versus flight time.

        ``y`` is a numeric field from ``history_fields()``; ``wrt`` must be an
        enabled continuous sweep parameter. Additional keyword options:
            at: Sweep-coordinate value; plot nearest available derivative
                curve only. None (default) plots all available curves.
            title, xlabel, ylabel: Custom title and axis labels.
            figsize: Figure size in inches, default (8, 5).
            grid, legend, legend_title: Grid/legend formatting.
            xlim, ylim: Bounds in displayed units. xscale, yscale: Positive
                        divisors for plotted values or 'linear'/'log'/etc.
            filters: Hold other full_grid sweep parameters fixed.
            include_infeasible: Include infeasible cases with histories.
            save: None/False (no file), True (auto PNG), or filename/path.
            filename, file_format: Custom output name and format; see plot().
            colorbar: None auto-enables for >8 curves; True/False override.
            cmap, colorbar_label, color_scale: Numeric colormap options.
            Numeric xscale/yscale divide plotted time/derivative values without
            changing the derivative calculation or colorbar's color_scale.
            One derivative curve defaults to black.
            show: True/False for interactive display; None is automatic.
            dpi: Image DPI, default 160; ax: existing matplotlib Axes.
            **style: Matplotlib line styling such as color, linewidth, alpha.

        Returns:
            matplotlib.figure.Figure: Figure for further customization.
        """
        from .plotting import plot_history_derivative
        return plot_history_derivative(self, y, wrt, **kwargs)
