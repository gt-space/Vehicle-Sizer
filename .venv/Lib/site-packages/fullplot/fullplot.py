from __future__ import annotations

import builtins
import itertools
import posixpath
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import butter, medfilt, savgol_filter, sosfiltfilt

from fullplot.hdf5 import hdf5_filename, safe_group_name
from fullplot.themes import (
    apply_theme,
    check_theme,
    grid_kwargs,
    style_colorbar,
    style_legend,
    theme_colors,
    theme_settings,
)


class FullPlotError(Exception):
    """Base exception for FullPlot errors."""


class DatasetNotFoundError(FullPlotError):
    """Raised when a dataset or group cannot be found."""


class AmbiguousDatasetError(FullPlotError):
    """Raised when a short dataset or group name matches more than one object."""


class PlotDataError(FullPlotError):
    """Raised when selected data cannot be plotted."""


VALID_TRACE_ROLES = ("data", "redline", "blueline", "yellowline", "greenline", "command")


def _normalize_trace_role(role: str | None) -> str:
    role = str(role or "data").lower().strip()

    if role not in VALID_TRACE_ROLES:
        valid = ", ".join(repr(item) for item in VALID_TRACE_ROLES)
        raise ValueError(f"Invalid trace role {role!r}. Valid roles are: {valid}.")

    return role


@dataclass
class LineSeries:
    """Internal description of one plotted line series.

    ``LineSeries`` is created after HDF5 selection, slicing, and trace expansion
    have reduced user inputs to one-dimensional arrays ready for Matplotlib. It
    is kept public-looking only because it is a small dataclass in this module;
    users normally interact with :class:`Trace` and :class:`H5File` instead.
    """

    data: np.ndarray
    label: str
    length: int
    x: np.ndarray | None = None
    role: str = "data"


def _as_1d_numeric_array(values, name: str, omit_nonfinite: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise PlotDataError(f"{name} must be one-dimensional.")

    if array.size == 0:
        raise PlotDataError(f"{name} cannot be empty.")

    if omit_nonfinite:
        array = array[np.isfinite(array)]

        if array.size == 0:
            raise PlotDataError(f"{name} has no finite values.")

    return array


def _omit_nonfinite_xy(x, y, name: str) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)

    if x_array.ndim != 1 or y_array.ndim != 1:
        raise PlotDataError(f"{name} must be one-dimensional after selection.")

    if x_array.shape != y_array.shape:
        raise PlotDataError(
            f"{name} has mismatched x and y lengths: "
            f"{len(x_array)} and {len(y_array)}."
        )

    # Non-finite time values are invalid, but non-finite y values are useful
    # information for engineering test data. Preserve NaN y samples so
    # Matplotlib draws visible gaps instead of connecting through dropout
    # regions.
    mask = np.isfinite(x_array)

    if not np.any(mask):
        raise PlotDataError(f"{name} has no finite x values to plot.")

    return x_array[mask], y_array[mask]


def _odd_window_count(count: int, minimum: int = 3) -> int:
    count = int(count)
    count = max(count, minimum)

    if count % 2 == 0:
        count += 1

    return count


def _trace_window_count(x: np.ndarray, window, minimum: int = 3) -> int:
    if window is None:
        raise ValueError("filter window must be provided for this filter.")

    if isinstance(window, (int, np.integer)):
        return _odd_window_count(int(window), minimum=minimum)

    window = float(window)

    if window <= 0.0:
        raise ValueError("filter window must be positive.")

    if x.size < 2:
        return _odd_window_count(1, minimum=minimum)

    dx = np.median(np.diff(x))

    if dx <= 0.0 or not np.isfinite(dx):
        raise ValueError("time-based filter windows require a strictly increasing x array.")

    return _odd_window_count(round(window / dx), minimum=minimum)


def _role_color(role: str, theme: str):
    role = str(role or "data").lower().strip()
    theme = check_theme(theme)

    role_colors = {
        "dark": {
            "redline": "#ff7a7a",
            "blueline": "#7ab8ff",
            "yellowline": "#ffd36a",
            "greenline": "#6aff9a",
        },
        "light": {
            "redline": "#7f1d1d",
            "blueline": "#245a9a",
            "yellowline": "#a87900",
            "greenline": "#1c6b3f",
        },
    }

    return role_colors[theme].get(role)


def _line_style_for_role(role: str, theme: str, color, linewidth: float) -> dict:
    role = _normalize_trace_role(role)
    line_color = _role_color(role, theme)

    if role == "command":
        return {
            "color": color,
            "linestyle": "-.",
            "linewidth": linewidth,
            "drawstyle": "steps-post",
        }

    if role in {"redline", "blueline", "yellowline", "greenline"}:
        return {
            "color": line_color,
            "linestyle": "--",
            "linewidth": 1.15 * linewidth,
            "alpha": 0.95,
        }

    return {
        "color": color,
        "linestyle": "-",
        "linewidth": linewidth,
    }


@dataclass
class TimeAxis:
    """Shared, shiftable and windowable time basis for related traces.

    Several ``Trace`` objects can reference one ``TimeAxis``. Shifting or
    windowing the axis updates all of those traces together without changing the
    original source arrays.
    """

    raw: np.ndarray
    name: str = "time"
    zero: float = 0.0
    attrs: dict[str, Any] | None = None

    def __post_init__(self):
        self.raw = _as_1d_numeric_array(self.raw, "TimeAxis raw time")

        if not np.all(np.isfinite(self.raw)):
            raise PlotDataError("TimeAxis cannot contain non-finite time values.")

        if self.raw.size > 1 and np.any(np.diff(self.raw) <= 0.0):
            raise PlotDataError("TimeAxis values must be strictly increasing.")

        self.name = str(self.name)
        self.zero = float(self.zero)
        self.attrs = {} if self.attrs is None else dict(self.attrs)
        self._window_start: float | None = None
        self._window_stop: float | None = None

    @property
    def full_values(self) -> np.ndarray:
        """Return all shifted time values before any active window is applied."""
        return self.raw - self.zero

    def _selection_for(self, start: float | None, stop: float | None) -> slice:
        if start is not None:
            start = float(start)
        if stop is not None:
            stop = float(stop)
        if start is not None and stop is not None and start > stop:
            raise ValueError("TimeAxis window start cannot be greater than stop.")

        values = self.full_values
        tol = 1e-12
        if self.raw.size > 1:
            tol = max(tol, abs(float(np.median(np.diff(self.raw)))) * 1e-9)

        left = 0 if start is None else int(np.searchsorted(values, start - tol, side="left"))
        right = len(values) if stop is None else int(np.searchsorted(values, stop + tol, side="right"))

        if left >= right:
            raise PlotDataError("TimeAxis window contains no samples.")

        return slice(left, right)

    @property
    def selection(self) -> slice:
        """Slice selecting the active samples from the full time axis."""
        return self._selection_for(self._window_start, self._window_stop)

    @property
    def values(self) -> np.ndarray:
        """Return shifted time values inside the active window."""
        return self.full_values[self.selection]

    @property
    def value(self) -> np.ndarray:
        """Alias for :attr:`values`."""
        return self.values

    @property
    def time(self) -> np.ndarray:
        """Alias for :attr:`values`."""
        return self.values

    @property
    def has_window(self) -> bool:
        """Whether an active time window is limiting the axis."""
        return self._window_start is not None or self._window_stop is not None

    @property
    def window_range(self) -> tuple[float | None, float | None]:
        """Current ``(start, stop)`` window in shifted time coordinates."""
        return self._window_start, self._window_stop

    @property
    def count(self) -> int:
        """Number of samples in the active time window."""
        return int(self.values.size)

    @property
    def raw_count(self) -> int:
        """Number of samples in the original time axis."""
        return int(self.raw.size)

    @property
    def tmin(self) -> float:
        """First shifted time value in the active window."""
        return float(self.values[0])

    @property
    def tmax(self) -> float:
        """Last shifted time value in the active window."""
        return float(self.values[-1])

    @property
    def raw_tmin(self) -> float:
        """First unshifted source time value."""
        return float(self.raw[0])

    @property
    def raw_tmax(self) -> float:
        """Last unshifted source time value."""
        return float(self.raw[-1])

    @property
    def duration(self) -> float:
        """Time span covered by the active window."""
        if self.count < 2:
            return 0.0
        return float(self.values[-1] - self.values[0])

    @property
    def raw_duration(self) -> float:
        """Time span covered by the original source axis."""
        if self.raw.size < 2:
            return 0.0
        return float(self.raw[-1] - self.raw[0])

    @property
    def dt_array(self) -> np.ndarray:
        """Adjacent spacings inside the active window."""
        return np.diff(self.values)

    @property
    def raw_dt_array(self) -> np.ndarray:
        """Adjacent spacings in the original source axis."""
        return np.diff(self.raw)

    @property
    def dt(self) -> float:
        """Nominal timestep in the active window."""
        if self.count < 2:
            return float("nan")
        return float(np.median(self.dt_array))

    @property
    def is_uniform(self) -> bool:
        """Whether active samples are approximately uniformly spaced."""
        if self.count < 3:
            return True
        dt = self.dt_array
        return bool(np.allclose(dt, np.median(dt), rtol=1e-6, atol=1e-12))

    def zero_at(self, time: float) -> "TimeAxis":
        """Shift the time basis so ``time`` in raw data appears as ``t = 0``."""
        zero = float(time)
        if zero != self.zero:
            old_zero = self.zero
            self.zero = zero
            try:
                self._selection_for(self._window_start, self._window_stop)
            except (PlotDataError, ValueError):
                self.zero = old_zero
                raise
        return self

    def align(self, data_time: float, model_time: float = 0.0) -> "TimeAxis":
        """Shift the axis so one raw data time aligns with one model time."""
        return self.zero_at(float(data_time) - float(model_time))

    def window(self, start: float | None = None, stop: float | None = None) -> "TimeAxis":
        """Limit this shared axis, and all attached traces, to a time interval."""
        start = None if start is None else float(start)
        stop = None if stop is None else float(stop)
        self._selection_for(start, stop)

        if (start, stop) != self.window_range:
            self._window_start = start
            self._window_stop = stop
        return self

    def clear_window(self) -> "TimeAxis":
        """Restore the full time range."""
        if self.has_window:
            self._window_start = None
            self._window_stop = None
        return self

    def windowed(self, start: float | None = None, stop: float | None = None) -> "TimeAxis":
        """Return an independent copy with a time window applied."""
        return self.copy().window(start=start, stop=stop)

    def copy(self, name: str | None = None) -> "TimeAxis":
        """Return an independent copy preserving alignment and window state."""
        copied = TimeAxis(
            raw=self.raw.copy(),
            name=self.name if name is None else name,
            zero=self.zero,
            attrs=self.attrs,
        )
        copied._window_start = self._window_start
        copied._window_stop = self._window_stop
        return copied

    def __len__(self) -> int:
        return self.count

    def __array__(self, dtype=None):
        array = self.values
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def __getitem__(self, item):
        return self.values[item]

class Trace:
    """Prepared one-dimensional engineering data series.

    A ``Trace`` pairs one x-array with one y-array and enough metadata to make
    the line reusable outside the original HDF5 file. It can represent raw test
    data, simulation output, filtered data, redlines, bluelines, yellowlines,
    greenlines, command traces, derived traces, or any other one-dimensional
    quantity. FullPlot roles only affect plotting style; they do not implement
    controller logic, abort logic, or unit conversion.

    Parameters
    ----------
    x:
        One-dimensional x values or a :class:`TimeAxis`. When a ``TimeAxis`` is
        supplied, the trace keeps a reference to it so alignment and windowing
        changes apply to every trace sharing that axis.
    y:
        One-dimensional y values. Non-finite y-values are preserved so missing
        test-data samples appear as plot gaps instead of being silently removed.
    name:
        Display name used in legends and HDF5 trace-writing output.
    role:
        Plotting role. Supported roles are ``"data"``, ``"redline"``,
        ``"blueline"``, ``"yellowline"``, ``"greenline"``, and ``"command"``.
    attrs:
        Optional user metadata. FullPlot stores this metadata when traces are
        written with :func:`write_traces`.
    """

    def __init__(
        self,
        x,
        y,
        name: str = "trace",
        role: str = "data",
        attrs: dict[str, Any] | None = None,
    ):
        self.time_axis = x if isinstance(x, TimeAxis) else None

        if self.time_axis is None:
            self._x = _as_1d_numeric_array(x, "Trace x")
        else:
            self._x = None

        y_array = _as_1d_numeric_array(y, "Trace y")

        if self.time_axis is None:
            if self._x.shape != y_array.shape:
                raise PlotDataError(
                    f"Trace x and y must have the same length. "
                    f"Received {len(self._x)} and {len(y_array)}."
                )
            self._raw_y = y_array

            mask = np.isfinite(self._x)
            if not np.any(mask):
                raise PlotDataError("Trace has no finite x values.")
            if not np.all(mask):
                self._x = self._x[mask]
                self._raw_y = self._raw_y[mask]
        else:
            if len(y_array) == self.time_axis.raw_count:
                self._raw_y = y_array
            elif len(y_array) == self.time_axis.count:
                self._raw_y = np.full(self.time_axis.raw_count, np.nan, dtype=float)
                self._raw_y[self.time_axis.selection] = y_array
            else:
                raise PlotDataError(
                    f"Trace y must match the TimeAxis raw length ({self.time_axis.raw_count}) "
                    f"or active length ({self.time_axis.count}). Received {len(y_array)}."
                )

        self.name = str(name)
        self.role = _normalize_trace_role(role)
        self.attrs = {} if attrs is None else dict(attrs)

    @property
    def x(self) -> np.ndarray:
        """Current x-values, including shared alignment and windowing."""
        if self.time_axis is not None:
            return self.time_axis.values
        return self._x

    @property
    def y(self) -> np.ndarray:
        """Current y-values, following any shared ``TimeAxis`` window."""
        if self.time_axis is not None:
            return self._raw_y[self.time_axis.selection]
        return self._raw_y

    @property
    def raw_y(self) -> np.ndarray:
        """Original y-array before shared ``TimeAxis`` windowing."""
        return self._raw_y

    @property
    def time(self) -> np.ndarray:
        """Alias for :attr:`x` when the trace represents time-series data."""
        return self.x

    @property
    def value(self) -> np.ndarray:
        """Alias for the trace y-values."""
        return self.y

    @property
    def finite(self) -> np.ndarray:
        """Boolean mask where both x and y are finite."""
        return np.isfinite(self.x) & np.isfinite(self.y)

    @property
    def tmin(self) -> float:
        """First x-value in the current trace coordinate system."""
        return float(self.x[0])

    @property
    def tmax(self) -> float:
        """Last x-value in the current trace coordinate system."""
        return float(self.x[-1])

    @property
    def time_range(self) -> tuple[float, float]:
        """Tuple of ``(tmin, tmax)`` for quick range checks."""
        return self.tmin, self.tmax

    def _x_for_new_trace(self):
        if self.time_axis is not None:
            return self.time_axis
        return self.x.copy()

    def _new(
        self,
        *,
        x=None,
        y=None,
        name: str | None = None,
        role: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> "Trace":
        """
        Create a new Trace while preserving user-facing metadata by default.

        FullPlot operations should not invent new display names. If the user
        wants a new legend name, the user can pass name=... or call renamed(...).
        """
        x_values = self._x_for_new_trace() if x is None else x
        if y is None:
            y_values = self.raw_y.copy() if isinstance(x_values, TimeAxis) else self.y.copy()
        else:
            y_values = np.asarray(y, dtype=float)

        return Trace(
            x=x_values,
            y=y_values,
            name=self.name if name is None else name,
            role=self.role if role is None else role,
            attrs=dict(self.attrs) if attrs is None else dict(attrs),
        )

    def renamed(self, name: str) -> "Trace":
        """
        Return a copy of this trace with a new display name.
        """
        return self._new(name=name)

    @classmethod
    def from_arrays(cls, name: str, x, y, role: str = "data", **attrs) -> "Trace":
        """Create a trace directly from explicit x/y arrays.

        This is equivalent to calling ``Trace(x=x, y=y, name=name, ...)`` but
        reads more clearly in scripts that generate traces from NumPy arrays.
        Extra keyword arguments are stored as trace metadata.
        """
        return cls(x=x, y=np.asarray(y, dtype=float), name=name, role=role, attrs=attrs)

    @classmethod
    def constant(cls, name: str, x, y: float, role: str = "data", **attrs) -> "Trace":
        """Create a constant-valued trace over an existing x-axis.

        This is the usual helper for redlines, bluelines, yellowlines,
        greenlines, targets, setpoints, and other constant references. If ``x``
        is a ``TimeAxis``, the new trace shares that time axis.
        """
        if isinstance(x, TimeAxis):
            x_values = x
            y_array = np.full(x.raw_count, float(y), dtype=float)
        else:
            x_values = _as_1d_numeric_array(x, "Trace x", omit_nonfinite=True)
            y_array = np.full_like(x_values, float(y), dtype=float)
        return cls(x=x_values, y=y_array, name=name, role=role, attrs=attrs)

    @classmethod
    def from_points(cls, name: str, points, x=None, mode: str = "linear", role: str = "data", **attrs) -> "Trace":
        """Create a trace by interpolating or holding a list of ``(x, y)`` points.

        ``mode="linear"`` linearly interpolates between points.
        ``mode="previous"`` creates a previous-value hold, which is useful for
        commands, valve schedules, controller setpoints, and sequence states. If
        ``x`` is omitted, the point x-values are used directly.
        """
        points = np.asarray(points, dtype=float)

        if points.ndim != 2 or points.shape[1] != 2:
            raise PlotDataError("points must be an array-like sequence of (x, y) pairs.")

        finite_points = np.isfinite(points).all(axis=1)
        points = points[finite_points]

        if points.size == 0:
            raise PlotDataError("points must contain at least one finite (x, y) pair.")

        xp = points[:, 0]
        yp = points[:, 1]
        order = np.argsort(xp)
        xp = xp[order]
        yp = yp[order]

        if x is None:
            x_values = xp
            x_array = xp
        elif isinstance(x, TimeAxis):
            x_values = x
            x_array = x.full_values
        else:
            x_values = _as_1d_numeric_array(x, "Trace x", omit_nonfinite=True)
            x_array = x_values

        mode = str(mode).lower().strip()

        if mode == "linear":
            y_array = np.interp(x_array, xp, yp)
        elif mode in {"previous", "step", "steps-post", "hold"}:
            indices = np.searchsorted(xp, x_array, side="right") - 1
            indices = np.clip(indices, 0, len(yp) - 1)
            y_array = yp[indices]
        else:
            raise ValueError("mode must be 'linear' or 'previous'.")

        return cls(x=x_values, y=y_array, name=name, role=role, attrs=attrs)

    @classmethod
    def from_function(cls, name: str, x, function, role: str = "data", **attrs) -> "Trace":
        """Create a trace by evaluating ``function(x_array)``.

        The function should accept a NumPy array and return a same-length array
        or array-like object. This is useful for analytic references, synthetic
        simulation curves, and generated limits that are not constant.
        """
        if isinstance(x, TimeAxis):
            x_values = x
            x_array = x.full_values
        else:
            x_values = _as_1d_numeric_array(x, "Trace x", omit_nonfinite=True)
            x_array = x_values
        y_array = np.asarray(function(x_array), dtype=float)
        return cls(x=x_values, y=y_array, name=name, role=role, attrs=attrs)

    def __call__(
        self,
        x,
        *,
        method: str = "previous",
        bounds: str = "nan",
    ):
        """
        Sample the trace at one or more x-values.

        By default, this uses the previous available sample, which is useful for
        sampled test data and solver coupling.

        Examples
        --------
        value = trace(3.0)

        values = trace([0.0, 0.1, 0.2])

        value = trace(3.0, method="linear")
        value = trace(3.0, method="nearest")
        """
        return self.value_at(x, method=method, bounds=bounds)

    def value_at(
        self,
        x,
        *,
        method: str = "previous",
        bounds: str = "nan",
    ):
        """
        Return the trace value at one or more x-values.

        Parameters
        ----------
        x:
            Scalar x-value or array of x-values.

        method:
            "previous"
                Use the last sample at or before x.
            "linear"
                Linearly interpolate between samples.
            "nearest"
                Use the nearest sample.

        bounds:
            "nan"
                Return NaN outside the trace x-range.
            "clamp"
                Clamp to the first or last available sample.
            "raise"
                Raise an error outside the trace x-range.
        """
        x_data = np.asarray(self.x, dtype=float)
        y_data = np.asarray(self.y, dtype=float)

        finite_x = np.isfinite(x_data)
        x_data = x_data[finite_x]
        y_data = y_data[finite_x]

        if x_data.size == 0:
            raise ValueError(f"Trace {self.name!r} has no finite x-values.")

        query = np.asarray(x, dtype=float)
        scalar = query.ndim == 0
        query = np.atleast_1d(query)

        output = np.full(query.shape, np.nan, dtype=float)

        finite_query = np.isfinite(query)
        inside = finite_query & (query >= x_data[0]) & (query <= x_data[-1])

        if bounds == "raise":
            outside = finite_query & ~inside
            if np.any(outside):
                raise ValueError(
                    f"Requested x-value outside trace range for {self.name!r}. "
                    f"Trace range is {x_data[0]} to {x_data[-1]}."
                )
            sample_mask = finite_query
            sample_x = query

        elif bounds == "nan":
            sample_mask = inside
            sample_x = query

        elif bounds == "clamp":
            sample_mask = finite_query
            sample_x = np.clip(query, x_data[0], x_data[-1])

        else:
            raise ValueError("bounds must be 'nan', 'clamp', or 'raise'.")

        if method == "previous":
            indices = np.searchsorted(x_data, sample_x, side="right") - 1
            indices = np.clip(indices, 0, x_data.size - 1)
            output[sample_mask] = y_data[indices[sample_mask]]

        elif method == "linear":
            output[sample_mask] = np.interp(sample_x[sample_mask], x_data, y_data)

        elif method == "nearest":
            right = np.searchsorted(x_data, sample_x, side="left")
            right = np.clip(right, 0, x_data.size - 1)
            left = np.clip(right - 1, 0, x_data.size - 1)

            choose_right = np.abs(x_data[right] - sample_x) < np.abs(sample_x - x_data[left])
            indices = np.where(choose_right, right, left)

            output[sample_mask] = y_data[indices[sample_mask]]

        else:
            raise ValueError("method must be 'previous', 'linear', or 'nearest'.")

        if scalar:
            return float(output[0])

        return output

    def copy(self, name: str | None = None, role: str | None = None, copy_time_axis: bool = False, **attrs) -> "Trace":
        """Return a copy of the trace with optional name, role, and metadata changes.

        By default, a trace sharing a ``TimeAxis`` continues to share that same
        axis. Set ``copy_time_axis=True`` when the copied trace should have an
        independent zero offset.
        """
        merged_attrs = dict(self.attrs)
        merged_attrs.update(attrs)
        x_values = self._x_for_new_trace()

        if copy_time_axis and isinstance(x_values, TimeAxis):
            x_values = x_values.copy()

        return Trace(
            x=x_values,
            y=self.raw_y.copy() if isinstance(x_values, TimeAxis) else self.y.copy(),
            name=self.name if name is None else name,
            role=self.role if role is None else role,
            attrs=merged_attrs,
        )

    def window(self, start: float | None = None, stop: float | None = None, name: str | None = None) -> "Trace":
        """Return a trace with values outside the time window replaced by NaN.

        The time axis is not cropped. Keeping the full time axis makes partial
        test-data traces easy to plot and easy for a solver to interpret as
        valid data inside the window and missing data outside the window.
        """
        mask = np.ones_like(self.x, dtype=bool)

        x = np.asarray(self.x, dtype=float)

        if x.size > 1:
            dx = np.nanmedian(np.diff(x))
            tol = max(1e-12, abs(dx) * 1e-9)
        else:
            tol = 1e-12

        if start is not None:
            mask &= x >= float(start) - tol

        if stop is not None:
            mask &= x <= float(stop) + tol

        y = self.y.copy()
        y[~mask] = np.nan

        attrs = dict(self.attrs)
        attrs.update({"window_start": start, "window_stop": stop, "window_mode": "nan_mask"})

        return self._new(y=y, name=name, attrs=attrs)

    def omit_missing(self, name: str | None = None) -> "Trace":
        """Return a compact trace containing only finite x/y pairs."""
        mask = self.finite

        if not np.any(mask):
            raise PlotDataError(f"Trace {self.name!r} has no finite x/y pairs.")

        return Trace(
            x=self.x[mask],
            y=self.y[mask],
            name=self.name if name is None else name,
            role=self.role,
            attrs=self.attrs,
        )

    drop_missing = omit_missing

    def filter(
        self,
        method: str = "moving_average",
        window=None,
        order: int = 2,
        cutoff: float | None = None,
        name: str | None = None,
        role: str | None = None,
    ) -> "Trace":
        """Return a filtered copy of the trace.

        Supported methods are ``"moving_average"``, ``"median"``,
        ``"savgol"``/``"savitzky_golay"``, and ``"lowpass"``. ``window`` may
        be an integer sample count or a positive x-width for the windowed
        filters. ``cutoff`` is required for the low-pass filter and is expressed
        in cycles per x-unit, normally Hz when x is seconds. Missing y-values are
        omitted during filtering and restored as ``NaN`` in the returned trace.
        """
        method = str(method).lower().strip().replace("-", "_").replace(" ", "_")

        finite_self = self.omit_missing(name=self.name)

        if method in {"moving_average", "mean", "average"}:
            count = _trace_window_count(finite_self.x, window, minimum=1)

            if count <= 1:
                finite_y = finite_self.y.copy()
            else:
                left = count // 2
                right = count - 1 - left

                padded_y = np.pad(
                    finite_self.y,
                    pad_width=(left, right),
                    mode="edge",
                )

                kernel = np.ones(count, dtype=float) / count
                finite_y = np.convolve(padded_y, kernel, mode="valid")

        elif method in {"median", "median_filter"}:
            count = _trace_window_count(finite_self.x, window, minimum=3)
            finite_y = medfilt(finite_self.y, kernel_size=count)

        elif method in {"savgol", "savitzky_golay", "savitzky-golay"}:
            count = _trace_window_count(finite_self.x, window, minimum=int(order) + 2)
            if count <= order:
                count = _odd_window_count(order + 2, minimum=order + 2)
            if count > len(finite_self.y):
                count = _odd_window_count(len(finite_self.y) if len(finite_self.y) % 2 else len(finite_self.y) - 1, minimum=3)
            finite_y = savgol_filter(finite_self.y, window_length=count, polyorder=int(order), mode="interp")

        elif method in {"lowpass", "low_pass"}:
            if cutoff is None:
                raise ValueError("lowpass filtering requires cutoff in Hz or cycles per x-unit.")
            if finite_self.x.size < 2:
                raise PlotDataError("lowpass filtering requires at least two samples.")
            dx = np.median(np.diff(finite_self.x))
            if dx <= 0.0 or not np.isfinite(dx):
                raise ValueError("lowpass filtering requires a strictly increasing x array.")
            fs = 1.0 / dx
            normalized_cutoff = float(cutoff) / (0.5 * fs)
            if not 0.0 < normalized_cutoff < 1.0:
                raise ValueError("lowpass cutoff must be between 0 and the Nyquist frequency.")
            sos = butter(N=4, Wn=normalized_cutoff, btype="lowpass", output="sos")
            finite_y = sosfiltfilt(sos, finite_self.y)

        else:
            raise ValueError("Unsupported filter method. Use moving_average, median, savgol, or lowpass.")

        if finite_self.y.size == self.y.size:
            y = np.asarray(finite_y, dtype=float)
        else:
            y = np.full_like(self.y, np.nan, dtype=float)
            y[self.finite] = finite_y

        attrs = dict(self.attrs)
        attrs.update({"filter": method, "window": window, "order": order, "cutoff": cutoff, "source": self.name})

        return self._new(
            y=np.asarray(y, dtype=float),
            name=name,
            role=role,
            attrs=attrs,
        )

    def scale(self, factor: float, name: str | None = None) -> "Trace":
        """Return a trace with all y-values multiplied by ``factor``."""
        return self._new(y=self.y * float(factor), name=name)

    def offset(self, value: float, name: str | None = None) -> "Trace":
        """Return a trace with ``value`` added to every y-sample."""
        return self._new(y=self.y + float(value), name=name)

    def derivative(self, name: str | None = None) -> "Trace":
        """Return a numerical derivative ``dy/dx`` trace using ``np.gradient``."""
        return self._new(y=np.gradient(self.y, self.x), name=name)

    def resample(self, x, name: str | None = None) -> "Trace":
        """Interpolate the trace onto a new x-array or ``TimeAxis``.

        Resampling uses ``np.interp`` and therefore performs one-dimensional
        linear interpolation. Values outside the original x-range are clamped by
        NumPy's default behavior. Use :meth:`value_at` when explicit out-of-bounds
        behavior is required.
        """
        if isinstance(x, TimeAxis):
            x_values = x
            x_array = x.full_values
        else:
            x_values = _as_1d_numeric_array(x, "resample x", omit_nonfinite=True)
            x_array = x_values

        return Trace(
            x=x_values,
            y=self.value_at(x_array, method="linear", bounds="nan"),
            name=self.name if name is None else name,
            role=self.role,
            attrs=self.attrs,
        )

    def _binary_operation(self, other, operation, symbol: str, reverse: bool = False) -> "Trace":
        attrs = dict(self.attrs)
        attrs.update({"operation": symbol})

        if isinstance(other, Trace):
            if reverse:
                left_y = other.resample(self.x).y
                right_y = self.y
                attrs.update({"left": other.name, "right": self.name})
            else:
                left_y = self.y
                right_y = other.resample(self.x).y
                attrs.update({"left": self.name, "right": other.name})

            return self._new(
                y=operation(left_y, right_y),
                attrs=attrs,
            )

        other_array = np.asarray(other)

        if reverse:
            y = operation(other_array, self.y)
        else:
            y = operation(self.y, other_array)

        attrs.update({"operand": other_array.item() if other_array.ndim == 0 else "array"})

        return self._new(
            y=y,
            attrs=attrs,
        )

    def __add__(self, other):
        return self._binary_operation(other, np.add, "+")

    def __radd__(self, other):
        return self._binary_operation(other, np.add, "+", reverse=True)

    def __sub__(self, other):
        return self._binary_operation(other, np.subtract, "-")

    def __rsub__(self, other):
        return self._binary_operation(other, np.subtract, "-", reverse=True)

    def __mul__(self, other):
        return self._binary_operation(other, np.multiply, "*")

    def __rmul__(self, other):
        return self._binary_operation(other, np.multiply, "*", reverse=True)

    def __truediv__(self, other):
        return self._binary_operation(other, np.divide, "/")

    def __rtruediv__(self, other):
        return self._binary_operation(other, np.divide, "/", reverse=True)


def _is_numeric_dtype(dtype) -> bool:
    try:
        return np.issubdtype(dtype, np.number)
    except TypeError:
        return False


def _shape_string(shape) -> str:
    if shape == ():
        return "scalar"

    return str(shape)


def _clean_root(root: str) -> str:
    root = str(root).strip()

    if root == "":
        root = "/"

    if not root.startswith("/"):
        root = "/" + root

    root = posixpath.normpath(root)

    if root == ".":
        root = "/"

    return root


def _join_h5(root: str, name: str) -> str:
    root = _clean_root(root)
    name = str(name).strip()

    if name.startswith("/"):
        return posixpath.normpath(name)

    if root == "/":
        return posixpath.normpath("/" + name)

    return posixpath.normpath(root + "/" + name)


def _basename(path: str) -> str:
    path = str(path).rstrip("/")

    if path == "":
        return ""

    return path.split("/")[-1]


def _relative_path(path: str, root: str) -> str:
    path = _clean_root(path)
    root = _clean_root(root)

    if root == "/":
        return path.lstrip("/")

    prefix = root.rstrip("/") + "/"

    if path.startswith(prefix):
        return path[len(prefix):]

    return path


def _normalize_name(name: str) -> str:
    name = str(name).lower().strip()

    for char in [" ", "\t", "\n", "\r", "_", "-", "[", "]", "(", ")", "{", "}", ".", ",", "/"]:
        name = name.replace(char, "")

    return name


def _as_list(value) -> builtins.list:
    if value is None:
        return []

    if isinstance(value, (builtins.list, tuple)):
        return builtins.list(value)

    return [value]


def _decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode(errors="replace")

    if isinstance(value, np.generic):
        return value.item()

    return value


def _check_scale(scale: str) -> str:
    scale = str(scale).lower().strip()

    if scale not in ("linear", "log"):
        raise ValueError("scale must be either 'linear' or 'log'.")

    return scale


def _check_log_positive(values, name: str):
    values = np.asarray(values)

    if values.size == 0:
        raise PlotDataError(f"{name} is empty and cannot be plotted on a log scale.")

    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        raise PlotDataError(f"{name} has no finite values and cannot be plotted on a log scale.")

    if np.any(finite_values <= 0.0):
        raise PlotDataError(f"{name} contains zero or negative values and cannot be plotted on a log scale.")


def _apply_plot_scales(ax, ax2, x_array, left_series, right_series, xscale, yscale, y2scale):
    xscale = _check_scale(xscale)
    yscale = _check_scale(yscale)
    y2scale = _check_scale(y2scale)

    all_series = builtins.list(left_series) + builtins.list(right_series)

    if xscale == "log":
        if x_array is not None:
            _check_log_positive(x_array, "x")

        for series in all_series:
            if series.x is not None:
                _check_log_positive(series.x, f"x for {series.label}")

    if yscale == "log":
        for series in left_series:
            _check_log_positive(series.data, series.label)

    if y2scale == "log":
        for series in right_series:
            _check_log_positive(series.data, series.label)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if ax2 is not None:
        ax2.set_yscale(y2scale)


def _normalize_axis(axis: int, ndim: int) -> int:
    axis = int(axis)

    if ndim <= 0:
        raise PlotDataError("Cannot select an axis from scalar data.")

    if axis < 0:
        axis += ndim

    if axis < 0 or axis >= ndim:
        raise PlotDataError(f"axis {axis} is out of range for array with {ndim} dimensions.")

    return axis


def _is_integer_index(value) -> bool:
    return isinstance(value, (int, np.integer))


def _normalize_slice_spec(slice_spec, ndim: int) -> dict[int, object]:
    if slice_spec is None:
        return {}

    if not isinstance(slice_spec, dict):
        raise PlotDataError("slice must be a dictionary such as {0: 3} or {2: (0, 10)}.")

    normalized = {}

    for axis, value in slice_spec.items():
        axis = int(axis)

        if axis < 0:
            axis += ndim

        if axis < 0 or axis >= ndim:
            raise PlotDataError(f"slice axis {axis} is out of range for array with {ndim} dimensions.")

        normalized[axis] = value

    return normalized


def _to_numpy_index(value):
    if isinstance(value, builtins.slice):
        return value

    if isinstance(value, tuple) or isinstance(value, builtins.list):
        if len(value) == 2:
            return builtins.slice(value[0], value[1])

        if len(value) == 3:
            return builtins.slice(value[0], value[1], value[2])

        raise PlotDataError("slice tuple/list values must have length 2 or 3.")

    return value


def _apply_slice_for_lines(array: np.ndarray, axis: int, slice_spec) -> tuple[np.ndarray, int]:
    array = np.asarray(array)

    if array.ndim == 0:
        raise PlotDataError("Cannot line-plot scalar data.")

    original_axis = _normalize_axis(axis, array.ndim)
    normalized_slice = _normalize_slice_spec(slice_spec, array.ndim)

    if original_axis in normalized_slice and _is_integer_index(normalized_slice[original_axis]):
        raise PlotDataError("The plotted axis cannot also be removed with an integer slice.")

    indexer = [builtins.slice(None)] * array.ndim

    for slice_axis, value in normalized_slice.items():
        indexer[slice_axis] = _to_numpy_index(value)

    sliced = array[tuple(indexer)]

    removed_before_axis = 0

    for slice_axis, value in normalized_slice.items():
        if slice_axis < original_axis and _is_integer_index(value):
            removed_before_axis += 1

    new_axis = original_axis - removed_before_axis

    return np.asarray(sliced), new_axis


def _apply_slice_for_map(array: np.ndarray, slice_spec) -> np.ndarray:
    array = np.asarray(array)

    if slice_spec is None:
        return array

    normalized_slice = _normalize_slice_spec(slice_spec, array.ndim)
    indexer = [builtins.slice(None)] * array.ndim

    for slice_axis, value in normalized_slice.items():
        indexer[slice_axis] = _to_numpy_index(value)

    return np.asarray(array[tuple(indexer)])


@dataclass
class H5File:
    """Handle for inspecting and plotting one HDF5 file or root group.

    Users normally create this object with :func:`open`. The object stores a
    filename and an HDF5 root path. Methods resolve short names, relative paths,
    and absolute paths under that root, then return trees, dataset lists, scalar
    values, reusable traces, line plots, or heat maps. The file is opened only
    for each operation, so the object is lightweight and safe to keep around in
    scripts.
    """

    filename: str | Path
    root: str = "/"

    def __post_init__(self):
        self.filename = str(self.filename)
        self.root = _clean_root(self.root)

    def at(self, group: str) -> "H5File":
        """
        Return a new H5File scoped to a specific HDF5 group.

        Example
        -------

        file = fplt.open("run.h5")
        run = file.at("/Pipe_Network/transient/runs/base")
        run.plot(x="time", y="tracks/Pipe_Mass_Flow_[kg_s]")
        """

        with h5py.File(self.filename, "r") as h5:
            group_path = self._resolve_group_path(h5, group)

        return H5File(self.filename, group_path)

    def tree(self, max_depth: int | None = None, print_output: bool = True) -> str:
        """Return a formatted tree of groups and datasets below the current root.

        Set ``print_output=False`` to collect the string without printing it.
        ``max_depth`` limits recursion depth and is useful for large files.
        """

        with h5py.File(self.filename, "r") as h5:
            if self.root not in h5:
                raise DatasetNotFoundError(f"Root path {self.root!r} was not found in {self.filename!r}.")

            lines = [f"{Path(self.filename).name}:{self.root}"]

            root_object = h5[self.root]

            if isinstance(root_object, h5py.Dataset):
                lines.append(self._dataset_tree_line(root_object, prefix="└── "))
            else:
                self._append_tree_lines(
                    lines=lines,
                    group=root_object,
                    prefix="",
                    depth=0,
                    max_depth=max_depth,
                )

        text = "\n".join(lines)

        if print_output:
            print(text)

        return text

    def list(self, print_output: bool = True) -> str:
        """Return a categorized list of datasets under the current root.

        Datasets are grouped into 1D numeric, 2D numeric, multidimensional
        numeric, scalar numeric, and non-numeric sections. This is the fastest
        way to decide whether to call ``plot()``, ``map()``, ``trace()``, or
        ``read()`` on a new file.
        """

        with h5py.File(self.filename, "r") as h5:
            paths = self._dataset_paths(h5)

            one_d = []
            two_d = []
            multi_d = []
            scalars = []
            non_numeric = []

            for path in paths:
                dataset = h5[path]
                rel = _relative_path(path, self.root)
                shape = dataset.shape
                dtype = dataset.dtype

                if not _is_numeric_dtype(dtype):
                    non_numeric.append((rel, shape, dtype))
                    continue

                if shape == ():
                    scalars.append((rel, shape, dtype))
                elif len(shape) == 1:
                    one_d.append((rel, shape, dtype))
                elif len(shape) == 2:
                    two_d.append((rel, shape, dtype))
                else:
                    multi_d.append((rel, shape, dtype))

        lines = [f"{Path(self.filename).name}:{self.root}", ""]

        if one_d:
            lines.append("1D numeric datasets:")
            lines.extend(self._format_dataset_rows(one_d))
            lines.append("")

        if two_d:
            lines.append("2D numeric datasets:")
            lines.extend(self._format_dataset_rows(two_d))
            lines.append("")

        if multi_d:
            lines.append("3D+ numeric datasets:")
            lines.extend(self._format_dataset_rows(multi_d))
            lines.append("")

        if scalars:
            lines.append("Scalar numeric datasets:")
            lines.extend(self._format_dataset_rows(scalars))
            lines.append("")

        if non_numeric:
            lines.append("Non-numeric datasets:")
            lines.extend(self._format_dataset_rows(non_numeric))
            lines.append("")

        text = "\n".join(lines).rstrip()

        if print_output:
            print(text)

        return text

    def values(self, group: str | None = None, print_output: bool = True) -> dict[str, Any]:
        """Read scalar datasets under the current root or a selected group.

        FullPlot treats scalar datasets as metadata-like values, such as a run
        setting, constant, or map attribute stored as a dataset. Non-scalar
        datasets are ignored by this method.
        """

        with h5py.File(self.filename, "r") as h5:
            if group is None:
                root = self.root
            else:
                root = self._resolve_group_path(h5, group)

            values = {}

            if isinstance(h5[root], h5py.Dataset):
                dataset = h5[root]

                if dataset.shape == ():
                    values[_basename(root)] = _decode_scalar(dataset[()])

            else:
                paths = self._dataset_paths(h5, root=root)

                for path in paths:
                    dataset = h5[path]

                    if dataset.shape == ():
                        rel = _relative_path(path, root)
                        values[rel] = _decode_scalar(dataset[()])

        if print_output:
            if values:
                width = max(len(name) for name in values)

                for name, value in values.items():
                    print(f"{name:<{width}}  {value}")
            else:
                print("No scalar datasets found.")

        return values

    def read(self, name: str):
        """Read one dataset by absolute path, relative path, or unique short name.

        The returned object is the raw NumPy value from ``h5py``. Use
        ``trace()`` or ``time()`` when the dataset should become a FullPlot
        object with validation and metadata.
        """

        with h5py.File(self.filename, "r") as h5:
            path = self._resolve_dataset_path(h5, name)
            return h5[path][()]

    def time(self, name: str = "time") -> TimeAxis:
        """Read a one-dimensional HDF5 dataset as a shared ``TimeAxis``.

        The returned axis can be passed to ``trace(..., x=time_axis)`` and later
        shifted, aligned, or windowed once for all attached traces.
        """

        with h5py.File(self.filename, "r") as h5:
            path = self._resolve_dataset_path(h5, name)
            dataset = h5[path]
            array = np.asarray(dataset[()])

            if array.ndim != 1:
                raise PlotDataError("time must be a 1D dataset.")

            if not _is_numeric_dtype(array.dtype):
                raise PlotDataError("time must be numeric.")

            attrs = {key: _decode_scalar(value) for key, value in dataset.attrs.items()}

        return TimeAxis(raw=array, name=_basename(path), attrs={"source_file": self.filename, "source_root": self.root, "source_dataset": path, **attrs})

    def traces(self) -> List[str]:
        """Return numeric one-dimensional dataset paths under the current root."""

        with h5py.File(self.filename, "r") as h5:
            names = []

            for path in self._dataset_paths(h5):
                dataset = h5[path]

                if dataset.shape != () and len(dataset.shape) == 1 and _is_numeric_dtype(dataset.dtype):
                    names.append(_relative_path(path, self.root))

        return names

    def trace(
        self,
        y,
        x=None,
        slice=None,
        axis: int = -1,
        name: str | None = None,
        role: str | None = None,
    ) -> Trace:
        """Read one HDF5 dataset or selected slice as a reusable ``Trace``.

        ``y`` may be a dataset path/name or an existing ``Trace``. ``x`` may be
        omitted, a dataset path/name, an array-like object, a ``Trace`` whose
        y-values are used as x-values, or a shared ``TimeAxis``. Multidimensional
        y-data must reduce to exactly one one-dimensional trace after applying
        ``slice`` and ``axis``.
        """

        if isinstance(y, Trace):
            return y.copy(name=name, role=role)

        with h5py.File(self.filename, "r") as h5:
            x_values = None
            x_array = None

            if x is not None:
                if isinstance(x, TimeAxis):
                    x_values = x
                    x_array = x.full_values
                elif isinstance(x, Trace):
                    x_values = x.y
                    x_array = x.y
                elif isinstance(x, str):
                    x_path = self._resolve_dataset_path(h5, x)
                    x_array = np.asarray(h5[x_path][()])
                    x_values = x_array
                else:
                    x_array = np.asarray(x)
                    x_values = x_array

                if x_array.ndim != 1:
                    raise PlotDataError("x must be a 1D dataset, array, TimeAxis, or Trace.")

                if not _is_numeric_dtype(x_array.dtype):
                    raise PlotDataError("x must be numeric.")

            series = self._build_line_series(
                h5=h5,
                selector=y,
                axis=axis,
                slice_spec=slice,
            )

        if len(series) != 1:
            raise PlotDataError(
                f"trace() expected one 1D trace, but {len(series)} traces were produced. "
                "Use plot() directly for multidimensional trace expansion."
            )

        series_item = series[0]

        if x_array is None:
            x_array = np.arange(series_item.length, dtype=float)
            x_values = x_array

        if len(x_array) != series_item.length:
            raise PlotDataError(
                f"Trace {series_item.label!r} has length {series_item.length}, "
                f"but x has length {len(x_array)}."
            )

        return Trace(
            x=x_values,
            y=series_item.data,
            name=series_item.label if name is None else name,
            role="data" if role is None else role,
            attrs={"source_file": self.filename, "source_root": self.root, "source_dataset": str(y)},
        )

    def _series_from_selector(self, h5: h5py.File, selector, axis: int, slice_spec) -> builtins.list[LineSeries]:
        if isinstance(selector, Trace):
            return [
                LineSeries(
                    data=selector.y,
                    label=selector.name,
                    length=len(selector.y),
                    x=selector.x,
                    role=selector.role,
                )
            ]

        return self._build_line_series(
            h5=h5,
            selector=selector,
            axis=axis,
            slice_spec=slice_spec,
        )

    def plot(
        self,
        y=None,
        x=None,
        y2=None,
        slice=None,
        axis: int = -1,
        labels=None,
        y2labels=None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        y2label: str | None = None,
        title: str | None = None,
        xscale: str = "linear",
        yscale: str = "linear",
        y2scale: str = "linear",
        xlim=None,
        ylim=None,
        y2lim=None,
        legend: bool = True,
        legend_location: str = "best",
        grid: bool = True,
        theme: str = "dark",
        save: str | Path | None = None,
        dpi: int = 200,
        show: bool = False,
        figsize=(9, 5),
        linewidth: float = 1.6,
    ):
        """Plot HDF5 datasets and/or ``Trace`` objects.

        ``y`` selects one or more left-axis traces, while ``y2`` selects optional
        right-axis traces. ``xlim``, ``ylim``, and ``y2lim`` change only the
        displayed figure range; they do not modify trace data. The function
        returns the Matplotlib figure and axes for further customization.
        """

        theme = check_theme(theme)

        if y is None and y2 is None:
            raise PlotDataError("plot requires y, y2, or both.")

        with h5py.File(self.filename, "r") if self.filename else nullcontext(None) as h5:
            x_array = None
            x_name = None
            x_time_axis = None

            if x is not None:
                if isinstance(x, Trace):
                    x_array = x.y
                    x_name = x.name
                elif isinstance(x, TimeAxis):
                    x_time_axis = x
                    x_array = x.values
                    x_name = x.name
                else:
                    if h5 is None:
                        x_array = np.asarray(x)
                        x_name = "x"
                    else:
                        x_path = self._resolve_dataset_path(h5, x)
                        x_name = _basename(x_path)
                        x_array = np.asarray(h5[x_path][()])

                if x_array.ndim != 1:
                    raise PlotDataError("x must be a 1D dataset, array, TimeAxis, or Trace.")

                if not _is_numeric_dtype(x_array.dtype):
                    raise PlotDataError("x must be numeric.")

            left_series = []
            right_series = []

            for selector in _as_list(y):
                left_series.extend(
                    self._series_from_selector(
                        h5=h5,
                        selector=selector,
                        axis=axis,
                        slice_spec=slice,
                    )
                )

            for selector in _as_list(y2):
                right_series.extend(
                    self._series_from_selector(
                        h5=h5,
                        selector=selector,
                        axis=axis,
                        slice_spec=slice,
                    )
                )

        all_series = left_series + right_series

        if not all_series:
            raise PlotDataError("No y data was selected.")
        if y2lim is not None and not right_series:
            raise PlotDataError("y2lim requires y2 data.")

        dataset_series = [series for series in all_series if series.x is None]

        if x_time_axis is not None:
            for series in dataset_series:
                if series.length == x_time_axis.raw_count:
                    series.data = series.data[x_time_axis.selection]
                    series.length = len(series.data)

        if x_array is None and dataset_series:
            x_length = dataset_series[0].length
            x_array = np.arange(x_length)
            x_name = "Index"
        elif x_array is not None:
            x_length = len(x_array)
        else:
            x_length = None
            x_name = "x"

        if x_array is not None:
            for series in dataset_series:
                if series.length != x_length:
                    raise PlotDataError(
                        f"Trace {series.label!r} has length {series.length}, "
                        f"but x has length {x_length}."
                    )

        for series in all_series:
            if series.x is not None:
                if len(series.x) != series.length:
                    raise PlotDataError(f"Trace {series.label!r} has mismatched x and y lengths.")
                if not _is_numeric_dtype(series.x.dtype):
                    raise PlotDataError(f"Trace {series.label!r} has non-numeric x data.")

        self._apply_user_labels(left_series, labels, "labels")
        self._apply_user_labels(right_series, y2labels, "y2labels")

        fig, ax = plt.subplots(figsize=figsize)

        ax2 = None

        if right_series:
            ax2 = ax.twinx()

        apply_theme(fig, [ax, ax2], theme)

        colors = itertools.cycle(theme_colors(theme))

        for series in left_series:
            color = next(colors)
            x_plot = series.x if series.x is not None else x_array
            x_clean, y_clean = _omit_nonfinite_xy(x_plot, series.data, series.label)
            style = _line_style_for_role(series.role, theme, color, linewidth)
            ax.plot(x_clean, y_clean, label=series.label, **style)

        if ax2 is not None:
            for series in right_series:
                color = next(colors)
                x_plot = series.x if series.x is not None else x_array
                x_clean, y_clean = _omit_nonfinite_xy(x_plot, series.data, series.label)
                style = _line_style_for_role(series.role, theme, color, linewidth)
                ax2.plot(x_clean, y_clean, label=series.label, **style)

        if xlabel is None:
            xlabel = x_name

        if ylabel is None:
            if len(left_series) == 1 and not right_series:
                ylabel = left_series[0].label
            else:
                ylabel = "Value"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if ax2 is not None:
            if y2label is None:
                if len(right_series) == 1:
                    y2label = right_series[0].label
                else:
                    y2label = "Value"

            ax2.set_ylabel(y2label)

        if title is not None:
            ax.set_title(title)

        _apply_plot_scales(
            ax=ax,
            ax2=ax2,
            x_array=x_array,
            left_series=left_series,
            right_series=right_series,
            xscale=xscale,
            yscale=yscale,
            y2scale=y2scale,
        )

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if y2lim is not None:
            ax2.set_ylim(y2lim)

        if grid:
            ax.grid(True, **grid_kwargs(theme))
        else:
            ax.grid(False)

        if legend:
            handles, legend_labels = ax.get_legend_handles_labels()

            if ax2 is not None:
                handles2, legend_labels2 = ax2.get_legend_handles_labels()
                handles += handles2
                legend_labels += legend_labels2

            if handles:
                legend_object = ax.legend(handles, legend_labels, loc=legend_location)
                style_legend(legend_object, theme)

        fig.tight_layout()

        self._save_and_show(fig=fig, save=save, dpi=dpi, show=show)

        if ax2 is not None:
            return fig, (ax, ax2)

        return fig, ax

    def map(
        self,
        z,
        x=None,
        y=None,
        slice=None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        title: str | None = None,
        xscale: str = "linear",
        yscale: str = "linear",
        zscale: str = "linear",
        grid: bool = False,
        theme: str = "dark",
        save: str | Path | None = None,
        dpi: int = 200,
        show: bool = True,
        figsize=(8, 6),
    ):
        """
        Plot a 2D dataset as a map.

        Parameters
        ----------
        z:
            One of the following:

            - A 2D dataset name/path.
            - A multidimensional dataset reduced to 2D with slice.
            - A list of 1D dataset names/paths, which will be stacked into
              a 2D array. This is useful for heat maps built from several
              separate time histories.

        x:
            Optional 1D x-axis dataset or array-like values.

        y:
            Optional 1D y-axis dataset or array-like values.

        slice:
            Optional integer-index slicing for 3D+ arrays.

        xscale, yscale, zscale:
            Axis and color scale options. Use "linear" or "log". A log scale
            only changes the displayed scale; it does not transform stored data.
        """

        theme = check_theme(theme)

        with h5py.File(self.filename, "r") as h5:
            z_array, z_name = self._read_map_z(h5, z)

            z_array = _apply_slice_for_map(z_array, slice)

            if z_array.ndim != 2:
                raise PlotDataError(
                    f"map requires 2D data after slicing. "
                    f"Selected data has shape {z_array.shape}."
                )

            rows, cols = z_array.shape

            x_array, x_name = self._read_map_axis(
                h5=h5,
                value=x,
                default=np.arange(cols),
                default_name="Column Index",
                valid_lengths=(cols, cols + 1),
                axis_name="x",
            )

            y_array, y_name = self._read_map_axis(
                h5=h5,
                value=y,
                default=np.arange(rows),
                default_name="Row Index",
                valid_lengths=(rows, rows + 1),
                axis_name="y",
            )

        fig, ax = plt.subplots(figsize=figsize)
        apply_theme(fig, ax, theme)

        xscale = _check_scale(xscale)
        yscale = _check_scale(yscale)
        zscale = _check_scale(zscale)

        if xscale == "log":
            _check_log_positive(x_array, "x")

        if yscale == "log":
            _check_log_positive(y_array, "y")

        norm = None

        if zscale == "log":
            _check_log_positive(z_array, "z")
            norm = LogNorm()

        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        cmap = theme_settings(theme)["map_cmap"]

        mesh = ax.pcolormesh(
            x_array,
            y_array,
            z_array,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        colorbar = fig.colorbar(mesh, ax=ax)
        style_colorbar(colorbar, theme)

        if xlabel is None:
            xlabel = x_name

        if ylabel is None:
            ylabel = y_name

        if zlabel is None:
            zlabel = z_name

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        colorbar.set_label(zlabel)

        if title is not None:
            ax.set_title(title)

        if grid:
            ax.grid(True, **grid_kwargs(theme))
        else:
            ax.grid(False)

        fig.tight_layout()

        self._save_and_show(fig=fig, save=save, dpi=dpi, show=show)

        return fig, ax

    def _resolve_dataset_path(self, h5: h5py.File, selector: str) -> str:
        if not isinstance(selector, str):
            raise DatasetNotFoundError(f"Dataset selector must be a string, got {type(selector)}.")

        selector = selector.strip()

        if selector == "":
            raise DatasetNotFoundError("Empty dataset selector.")

        direct_path = _join_h5(self.root, selector)

        if direct_path in h5:
            if isinstance(h5[direct_path], h5py.Dataset):
                return direct_path

            raise DatasetNotFoundError(f"{direct_path!r} exists, but it is not a dataset.")

        paths = self._dataset_paths(h5)

        exact_matches = []

        for path in paths:
            basename = _basename(path)
            relative = _relative_path(path, self.root)

            if selector == basename or selector == relative:
                exact_matches.append(path)

        if len(exact_matches) == 1:
            return exact_matches[0]

        if len(exact_matches) > 1:
            self._raise_ambiguous(selector, exact_matches)

        normalized_selector = _normalize_name(selector)
        normalized_matches = []

        for path in paths:
            basename = _basename(path)
            relative = _relative_path(path, self.root)

            if (
                _normalize_name(basename) == normalized_selector
                or _normalize_name(relative) == normalized_selector
            ):
                normalized_matches.append(path)

        if len(normalized_matches) == 1:
            return normalized_matches[0]

        if len(normalized_matches) > 1:
            self._raise_ambiguous(selector, normalized_matches)

        raise DatasetNotFoundError(
            f"Could not find dataset {selector!r} under {self.root!r} in {self.filename!r}."
        )

    def _resolve_group_path(self, h5: h5py.File, selector: str) -> str:
        selector = str(selector).strip()

        if selector == "":
            raise DatasetNotFoundError("Empty group selector.")

        direct_path = _join_h5(self.root, selector)

        if direct_path in h5:
            if isinstance(h5[direct_path], h5py.Group):
                return direct_path

            raise DatasetNotFoundError(f"{direct_path!r} exists, but it is not a group.")

        groups = self._group_paths(h5)

        matches = []

        for path in groups:
            basename = _basename(path)
            relative = _relative_path(path, self.root)

            if selector == basename or selector == relative:
                matches.append(path)

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            self._raise_ambiguous(selector, matches)

        raise DatasetNotFoundError(
            f"Could not find group {selector!r} under {self.root!r} in {self.filename!r}."
        )

    def _dataset_paths(self, h5: h5py.File, root: str | None = None) -> builtins.list[str]:
        if root is None:
            root = self.root

        root = _clean_root(root)

        if root not in h5:
            raise DatasetNotFoundError(f"Root path {root!r} was not found in {self.filename!r}.")

        root_object = h5[root]
        paths = []

        if isinstance(root_object, h5py.Dataset):
            return [root]

        def visit(name, object_):
            if isinstance(object_, h5py.Dataset):
                if root == "/":
                    paths.append("/" + name)
                else:
                    paths.append(root.rstrip("/") + "/" + name)

        root_object.visititems(visit)
        paths.sort()

        return paths

    def _group_paths(self, h5: h5py.File) -> builtins.list[str]:
        if self.root not in h5:
            raise DatasetNotFoundError(f"Root path {self.root!r} was not found in {self.filename!r}.")

        root_object = h5[self.root]
        paths = []

        if isinstance(root_object, h5py.Group):
            paths.append(self.root)

            def visit(name, object_):
                if isinstance(object_, h5py.Group):
                    if self.root == "/":
                        paths.append("/" + name)
                    else:
                        paths.append(self.root.rstrip("/") + "/" + name)

            root_object.visititems(visit)

        paths.sort()

        return paths

    def _raise_ambiguous(self, selector: str, matches: builtins.list[str]):
        lines = [f"Dataset/group name {selector!r} is ambiguous. Matches:"]

        for path in matches:
            lines.append(f"  - {path}")

        lines.append("Use a full path or scope the file with file.at(...).")

        raise AmbiguousDatasetError("\n".join(lines))

    def _dataset_tree_line(self, dataset: h5py.Dataset, prefix: str) -> str:
        name = _basename(dataset.name)
        shape = _shape_string(dataset.shape)
        dtype = dataset.dtype

        return f"{prefix}{name}  {shape}  {dtype}"

    def _append_tree_lines(
        self,
        lines: builtins.list[str],
        group: h5py.Group,
        prefix: str,
        depth: int,
        max_depth: int | None,
    ):
        if max_depth is not None and depth >= max_depth:
            return

        items = sorted(group.items(), key=lambda item: item[0])

        for index, (name, object_) in enumerate(items):
            is_last = index == len(items) - 1
            branch = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            if isinstance(object_, h5py.Dataset):
                shape = _shape_string(object_.shape)
                dtype = object_.dtype
                lines.append(f"{prefix}{branch}{name}  {shape}  {dtype}")
            else:
                lines.append(f"{prefix}{branch}{name}/")
                self._append_tree_lines(
                    lines=lines,
                    group=object_,
                    prefix=next_prefix,
                    depth=depth + 1,
                    max_depth=max_depth,
                )

    def _format_dataset_rows(self, rows: builtins.list[tuple[str, Any, Any]]) -> builtins.list[str]:
        if not rows:
            return []

        width = max(len(row[0]) for row in rows)
        formatted = []

        for name, shape, dtype in rows:
            formatted.append(f"  {name:<{width}}  {_shape_string(shape):<16}  {dtype}")

        return formatted

    def _build_line_series(
        self,
        h5: h5py.File,
        selector: str,
        axis: int,
        slice_spec,
    ) -> builtins.list[LineSeries]:
        path = self._resolve_dataset_path(h5, selector)
        base_label = _basename(path)

        array = np.asarray(h5[path][()])

        if not _is_numeric_dtype(array.dtype):
            raise PlotDataError(f"Dataset {path!r} is not numeric.")

        array, plot_axis = _apply_slice_for_lines(array, axis=axis, slice_spec=slice_spec)

        if array.ndim == 0:
            raise PlotDataError(f"Dataset {path!r} became scalar after slicing.")

        plot_axis = _normalize_axis(plot_axis, array.ndim)
        length = array.shape[plot_axis]

        moved = np.moveaxis(array, plot_axis, -1)

        if moved.ndim == 1:
            return [
                LineSeries(
                    data=np.asarray(moved),
                    label=base_label,
                    length=length,
                )
            ]

        trace_shape = moved.shape[:-1]
        flattened = moved.reshape((-1, moved.shape[-1]))
        series = []

        for trace_index, trace in enumerate(flattened):
            multi_index = np.unravel_index(trace_index, trace_shape)

            if len(multi_index) == 1:
                label = f"{base_label}[{multi_index[0]}]"
            else:
                label = f"{base_label}{multi_index}"

            series.append(
                LineSeries(
                    data=np.asarray(trace),
                    label=label,
                    length=length,
                )
            )

        return series

    def _apply_user_labels(self, series: builtins.list[LineSeries], labels, label_name: str):
        if labels is None:
            return

        labels = _as_list(labels)

        if len(labels) != len(series):
            raise PlotDataError(
                f"{label_name} must contain exactly {len(series)} labels. "
                f"Received {len(labels)}."
            )

        for series_item, label in zip(series, labels):
            series_item.label = str(label)

    def _save_and_show(self, fig, save, dpi: int, show: bool):
        if save is not None:
            fig.savefig(
                save,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )

        if show:
            plt.show()

    def _read_map_z(self, h5: h5py.File, z):
        if isinstance(z, (builtins.list, tuple)):
            arrays = []

            if len(z) == 0:
                raise PlotDataError("z cannot be an empty list.")

            for selector in z:
                path = self._resolve_dataset_path(h5, selector)
                array = np.asarray(h5[path][()])

                if not _is_numeric_dtype(array.dtype):
                    raise PlotDataError(f"Dataset {path!r} is not numeric.")

                if array.ndim != 1:
                    raise PlotDataError(
                        f"When z is a list, each selected dataset must be 1D. "
                        f"Dataset {path!r} has shape {array.shape}."
                    )

                arrays.append(array)

            lengths = {len(array) for array in arrays}

            if len(lengths) != 1:
                raise PlotDataError("All z datasets must have the same length to form a heat map.")

            return np.vstack(arrays), "Stacked Values"

        z_path = self._resolve_dataset_path(h5, z)
        z_name = _basename(z_path)
        z_array = np.asarray(h5[z_path][()])

        if not _is_numeric_dtype(z_array.dtype):
            raise PlotDataError("z must be numeric.")

        return z_array, z_name

    def _read_map_axis(self, h5: h5py.File, value, default, default_name: str, valid_lengths, axis_name: str):
        if value is None:
            return default, default_name

        if isinstance(value, str):
            path = self._resolve_dataset_path(h5, value)
            array = np.asarray(h5[path][()])
            name = _basename(path)
        else:
            array = np.asarray(value)
            name = default_name

        if array.ndim != 1:
            raise PlotDataError(f"{axis_name} must be 1D.")

        if not _is_numeric_dtype(array.dtype):
            raise PlotDataError(f"{axis_name} must be numeric.")

        if len(array) not in valid_lengths:
            valid_text = " or ".join(str(length) for length in valid_lengths)
            raise PlotDataError(
                f"{axis_name} has length {len(array)}, but expected length {valid_text}."
            )

        return array, name
        


def _resolve_hdf5_read_filename(filename: str | Path) -> str:
    """Return an existing HDF5 filename, trying .h5 when no suffix is given."""
    path = Path(filename)

    # Respect exact filenames first. This avoids breaking users who really do
    # have an extensionless HDF5 file.
    if path.exists():
        return str(path)

    # Convenience behavior:
    #
    #     fplt.open("run")
    #
    # should open "run.h5" if that is what FullFlow wrote.
    if path.suffix == "":
        h5_path = path.with_suffix(".h5")

        if h5_path.exists():
            return str(h5_path)

        # Return the .h5 candidate anyway so the eventual FileNotFoundError
        # points at the filename FullPlot actually tried to open.
        return str(h5_path)

    return str(path)


def open(filename: str | Path, root: str = "/") -> H5File:
    """Open an HDF5 file for FullPlot inspection and plotting.

    The file is not held open; the returned ``H5File`` stores the filename and
    root group and opens the file for each operation. If ``filename`` has no
    suffix, FullPlot first looks for an exact file and then tries ``.h5``.
    """
    return H5File(filename=_resolve_hdf5_read_filename(filename), root=root)


def tree(filename: str | Path, root: str = "/", **kwargs) -> str:
    """Convenience wrapper for ``open(filename, root).tree(...)``."""
    return open(filename, root=root).tree(**kwargs)


def list_h5(filename: str | Path, root: str = "/", **kwargs) -> str:
    """Convenience wrapper for ``open(filename, root).list(...)``."""
    return open(filename, root=root).list(**kwargs)


def read(filename: str | Path, name: str, root: str = "/"):
    """Read one HDF5 dataset without explicitly creating an ``H5File`` object."""
    return open(filename, root=root).read(name)


def time(filename: str | Path, name: str = "time", root: str = "/") -> TimeAxis:
    """Read one HDF5 time dataset as a shared ``TimeAxis``."""
    return open(filename, root=root).time(name)


def values(filename: str | Path, root: str = "/", group: str | None = None, **kwargs):
    """Read scalar datasets from a file, root, or selected group."""
    return open(filename, root=root).values(group=group, **kwargs)


def trace(filename: str | Path, y, x=None, root: str = "/", **kwargs) -> Trace:
    """Read one dataset from a file as a reusable ``Trace``."""
    return open(filename, root=root).trace(y=y, x=x, **kwargs)


def _looks_like_trace_collection(value) -> bool:
    if isinstance(value, Trace):
        return True

    if isinstance(value, (builtins.list, tuple)) and value:
        return all(isinstance(item, Trace) for item in value)

    return False


def plot(filename_or_traces=None, y=None, x=None, root: str = "/", **kwargs):
    """Plot either an HDF5 dataset selection or prepared Trace objects."""

    if _looks_like_trace_collection(filename_or_traces) and y is None:
        return H5File(filename="", root="/").plot(y=filename_or_traces, x=x, **kwargs)

    if filename_or_traces is None:
        raise TypeError("plot() requires either a filename or Trace object(s).")

    return open(filename_or_traces, root=root).plot(y=y, x=x, **kwargs)


def map(filename: str | Path, z, root: str = "/", **kwargs):
    """Plot a 2D HDF5 dataset or stacked 1D datasets as a heat map."""
    return open(filename, root=root).map(z=z, **kwargs)


def write_traces(filename: str | Path, traces, group: str = "traces", overwrite: bool = True) -> str:
    """Write one or more Trace objects to a simple HDF5 trace group."""

    traces = _as_list(traces)

    if not traces:
        raise ValueError("write_traces() requires at least one Trace.")

    for item in traces:
        if not isinstance(item, Trace):
            raise TypeError("write_traces() only accepts Trace objects.")

    output = hdf5_filename(filename)

    with h5py.File(output, "a") as h5:
        root_group = h5.require_group(safe_group_name(group))

        for item in traces:
            name = safe_group_name(item.name)

            if name in root_group:
                if overwrite:
                    del root_group[name]
                else:
                    raise ValueError(f"Trace group {group}/{name} already exists.")

            trace_group = root_group.create_group(name)
            trace_group.create_dataset("x", data=item.x)
            trace_group.create_dataset("y", data=item.y)
            trace_group.attrs["name"] = item.name
            trace_group.attrs["role"] = item.role

            for key, value in item.attrs.items():
                try:
                    trace_group.attrs[str(key)] = value
                except TypeError:
                    trace_group.attrs[str(key)] = str(value)

    return output


def show():
    """Show all currently open matplotlib figures."""

    plt.show()


list = list_h5



__all__ = [
    "FullPlotError",
    "DatasetNotFoundError",
    "AmbiguousDatasetError",
    "PlotDataError",
    "TimeAxis",
    "Trace",
    "H5File",
    "open",
    "tree",
    "list",
    "read",
    "values",
    "time",
    "trace",
    "plot",
    "map",
    "write_traces",
    "show",
]
