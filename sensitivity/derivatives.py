"""Numerical derivatives on nonuniform sweep grids and synchronized flight times."""
from __future__ import annotations

import numpy as np


def gradient(x, y):
    """d(y)/d(x), central inside and one-sided at boundaries; nonuniform x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or len(x) < 2 or y.shape[0] != len(x):
        raise ValueError('Derivatives require at least two distinct x samples and matching output data')
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError('Derivative sweep coordinates must be finite, unique and strictly increasing')
    return np.gradient(y, x, axis=0, edge_order=2 if len(x) >= 3 else 1)


def history_gradient(samples: list[tuple[float, dict]], output: str):
    """Interpolate selected histories only over their common observed time window.

    Returns (x_values, time_s, dy_dx), with dy_dx shape (nx, nt).
    """
    if len(samples) < 2:
        raise ValueError('Need two histories with different sweep values')
    samples = sorted(samples, key=lambda item: item[0])
    x = np.array([item[0] for item in samples], dtype=float)
    histories = [item[1] for item in samples]
    starts = []
    ends = []
    for history in histories:
        if 'time_s' not in history or output not in history:
            raise KeyError(f'History is missing time_s or {output!r}')
        t = history['time_s']
        if len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError('Flight times must be strictly increasing, finite and have >=2 samples')
        starts.append(t[0])
        ends.append(t[-1])
    lo, hi = max(starts), min(ends)
    if hi <= lo:
        raise ValueError('Sweep histories have no overlapping time interval')
    # Build a common time axis from the highest-resolution history (no extrapolation).
    selected = max(histories, key=lambda h: np.count_nonzero((h['time_s'] >= lo) & (h['time_s'] <= hi)))
    times = np.unique(np.concatenate(([lo], selected['time_s'][(selected['time_s'] > lo) &
                                                             (selected['time_s'] < hi)], [hi])))
    values = np.stack([np.interp(times, history['time_s'], history[output])
                       for history in histories])
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{output} contains missing/nonfinite values in the common time range')
    return x, times, gradient(x, values)
