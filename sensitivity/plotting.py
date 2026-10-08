"""Matplotlib plotting API for stored sensitivity results.

Functions return a matplotlib.figure.Figure for further customization.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _filename(*parts: str) -> str:
    slug = '_'.join(re.sub(r'[^a-zA-Z0-9]+', '_', str(part)).strip('_')
                    for part in parts)
    return slug.lower() + '.png'


def _axes(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.figure, ax, False


def _finish(results, fig, ax, *, default_xlabel, default_ylabel, default_name,
            title=None, xlabel=None, ylabel=None, xlim=None, ylim=None,
            xscale='linear', yscale='linear', grid=True, legend=None,
            legend_title=None, save=None, show=None, dpi=160, owns_axes=True):
    """Apply common formatting, optionally save/show, and return the Figure.

    save=None/False: don't save. save=True: save under results.folder/plots.
    save=path: save at that path. show=None: display only for unsaved plots
    created by the method itself (not for user-supplied axes).
    """
    ax.set(xlabel=default_xlabel if xlabel is None else xlabel,
           ylabel=default_ylabel if ylabel is None else ylabel,
           xscale=xscale, yscale=yscale)
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(bool(grid), alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if legend is None:
        legend = bool(handles)
    if legend and handles:
        ax.legend(title=legend_title)
    elif not legend and ax.get_legend() is not None:
        ax.get_legend().remove()

    if owns_axes:
        fig.tight_layout()

    if save is True:
        destination = Path(results.folder) / 'plots' / default_name
    elif save is None or save is False:
        destination = None
    elif isinstance(save, (str, Path)):
        destination = Path(save).expanduser()
    else:
        raise TypeError('save must be None, bool, str, or pathlib.Path')

    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=dpi, bbox_inches='tight')

    if show is None:
        show = destination is None and owns_axes
    if show:
        plt.show()
    return fig


def _plot_line_or_scatter(ax, x, y, *, kind, default_kind='line', style=None):
    style = dict(style or {})
    if kind == 'auto':
        kind = default_kind
    if kind == 'line':
        style.setdefault('marker', 'o')
        ax.plot(x, y, **style)
    elif kind == 'scatter':
        ax.scatter(x, y, **style)
    else:
        raise ValueError("kind must be 'auto', 'line', or 'scatter'")


def plot_scalar(results, x: str, y: str, *, filters=None, include_infeasible=False,
                kind='auto', title=None, xlabel=None, ylabel=None,
                figsize=(8, 5), grid=True, legend=None, legend_title=None,
                xlim=None, ylim=None, xscale='linear', yscale='linear',
                save=None, show=None, dpi=160, ax=None, **style):
    """Plot two scalar results; x can be a sweep input or another output.

    kind='auto' selects a line for sweep-input x, scatter for output x.
    Extra style keywords are passed to matplotlib's plot/scatter, e.g.
    color='navy', marker='s', linewidth=2, label='My design'.
    """
    points = results.scalar_points(x, y, filters=filters,
                                   include_infeasible=include_infeasible)
    if not points:
        raise ValueError(f'No valid scalar data for x={x!r}, y={y!r}; '
                         f'available: {results.fields()}')
    fig, axes, owns_axes = _axes(ax, figsize)
    try:
        _plot_line_or_scatter(axes, [p[0] for p in points], [p[1] for p in points],
                              kind=kind,
                              default_kind='line' if x in results.parameters else 'scatter',
                              style=style)
    except Exception:
        if owns_axes:
            plt.close(fig)
        raise
    return _finish(results, fig, axes, default_xlabel=x, default_ylabel=y,
                   default_name=_filename('scalar', y, 'vs', x),
                   title=title, xlabel=xlabel, ylabel=ylabel,
                   grid=grid, legend=legend, legend_title=legend_title,
                   xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale,
                   save=save, show=show, dpi=dpi, owns_axes=owns_axes)


def plot_history(results, y: str, group_by: str, *, filters=None,
                 include_infeasible=False, title=None, xlabel=None, ylabel=None,
                 figsize=(8, 5), grid=True, legend=None, legend_title=None,
                 xlim=None, ylim=None, xscale='linear', yscale='linear',
                 save=None, show=None, dpi=160, ax=None, case_labels=None, **style):
    """Plot recorded flight histories for cases varying one sweep parameter.

    case_labels optionally maps a case ID or numeric sweep value to a custom
    legend label. Extra style keywords are forwarded to ax.plot().
    """
    if group_by not in results.parameters:
        raise ValueError(f'{group_by!r} must be an enabled sweep parameter')
    cases = results._select(varying=group_by, filters=filters,
                            include_baseline=results.mode == 'one_at_a_time',
                            include_infeasible=include_infeasible)
    fig, axes, owns_axes = _axes(ax, figsize)
    plotted = 0
    for row in cases:
        try:
            data = results.history(row['case_id'])
        except KeyError:
            continue
        if y not in data or 'time_s' not in data:
            continue
        label = 'baseline' if row['is_baseline'] else f'{group_by}={row[group_by]:g}'
        if case_labels is not None:
            label = case_labels.get(row['case_id'], case_labels.get(row.get(group_by), label))
        line_style = dict(style)
        line_style.setdefault('label', label)
        axes.plot(data['time_s'], data[y], **line_style)
        plotted += 1
    if not plotted:
        if owns_axes:
            plt.close(fig)
        raise ValueError(f'No histories with output {y!r}; available: {results.history_fields()}')
    return _finish(results, fig, axes, default_xlabel='time_s', default_ylabel=y,
                   default_name=_filename('history', y, 'by', group_by),
                   title=title, xlabel=xlabel, ylabel=ylabel,
                   grid=grid, legend=legend, legend_title=legend_title,
                   xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale,
                   save=save, show=show, dpi=dpi, owns_axes=owns_axes)


def plot_derivative(results, x: str, y: str, wrt: str, *, filters=None,
                    include_infeasible=False, kind='line', title=None,
                    xlabel=None, ylabel=None, figsize=(8, 5), grid=True,
                    legend=None, legend_title=None, xlim=None, ylim=None,
                    xscale='linear', yscale='linear', save=None, show=None,
                    dpi=160, ax=None, **style):
    """Plot a scalar partial derivative dy/d(wrt) against x."""
    values, dydx = results.scalar_derivative(y, wrt, filters=filters,
                                              include_infeasible=include_infeasible)
    if x == wrt:
        abscissa = values
    else:
        cases = results._select(varying=wrt, filters=filters,
                                include_baseline=results.mode == 'one_at_a_time',
                                include_infeasible=include_infeasible)
        coordinates = {row[wrt]: row.get(x) for row in cases}
        if any(not isinstance(coordinates.get(v), (float, int)) for v in values):
            raise ValueError(f'Cannot use {x!r} as derivative plot x-axis')
        abscissa = [coordinates[v] for v in values]
    fig, axes, owns_axes = _axes(ax, figsize)
    try:
        _plot_line_or_scatter(axes, abscissa, dydx, kind=kind, style=style)
    except Exception:
        if owns_axes:
            plt.close(fig)
        raise
    return _finish(results, fig, axes, default_xlabel=x,
                   default_ylabel=f'd({y})/d({wrt})',
                   default_name=_filename('derivative', y, 'wrt', wrt, 'vs', x),
                   title=title, xlabel=xlabel, ylabel=ylabel,
                   grid=grid, legend=legend, legend_title=legend_title,
                   xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale,
                   save=save, show=show, dpi=dpi, owns_axes=owns_axes)


def plot_history_derivative(results, y: str, wrt: str, *, filters=None,
                            include_infeasible=False, at=None,
                            title=None, xlabel=None, ylabel=None,
                            figsize=(8, 5), grid=True, legend=None,
                            legend_title=None, xlim=None, ylim=None,
                            xscale='linear', yscale='linear',
                            save=None, show=None, dpi=160, ax=None, **style):
    """Plot d(history y)/d(wrt) versus flight time.

    at selects the nearest available sweep-coordinate derivative; otherwise
    show one derivative curve for each available sweep point.
    """
    xs, times, derivative = results.history_derivative(
        y, wrt, filters=filters, include_infeasible=include_infeasible)
    fig, axes, owns_axes = _axes(ax, figsize)
    if at is None:
        indices = range(len(xs))
    else:
        indices = [int(np.argmin(abs(xs - float(at))))]
    for index in indices:
        line_style = dict(style)
        line_style.setdefault('label', f'{wrt}={xs[index]:g}')
        axes.plot(times, derivative[index], **line_style)
    return _finish(results, fig, axes, default_xlabel='time_s',
                   default_ylabel=f'd({y})/d({wrt})',
                   default_name=_filename('history_derivative', y, 'wrt', wrt),
                   title=title, xlabel=xlabel, ylabel=ylabel,
                   grid=grid, legend=legend, legend_title=legend_title,
                   xlim=xlim, ylim=ylim, xscale=xscale, yscale=yscale,
                   save=save, show=show, dpi=dpi, owns_axes=owns_axes)
