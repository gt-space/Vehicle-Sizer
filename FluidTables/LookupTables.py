from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import h5py
import numpy as np
from errors import LookupBoundsError


def _json_attribute(attributes: h5py.AttributeManager, name: str) -> Any:
    if name not in attributes:
        raise ValueError(f"Lookup table is missing the '{name}' attribute")
    value = attributes[name]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Lookup attribute '{name}' is not valid JSON") from error


class LookupTable:
    """One rectangular-grid table loaded from an HDF5 file."""

    def __init__(
        self,
        name: str,
        axis_order: Tuple[str, ...],
        axes: Mapping[str, np.ndarray],
        outputs: Mapping[str, np.ndarray],
        axis_units: Mapping[str, str],
        output_units: Mapping[str, str],
        metadata: Mapping[str, Any],
        constants: Mapping[str, Any],
        status: Mapping[str, np.ndarray],
        discrete_axes: Iterable[str] = (),
    ) -> None:
        self.name = name
        self.axis_order = axis_order
        self.axes = dict(axes)
        self.output_names = tuple(outputs)
        self.output_index = {
            name: index for index, name in enumerate(self.output_names)
        }
        self._values = np.stack(
            [outputs[name] for name in self.output_names], axis=-1
        )
        self.outputs = {
            name: self._values[..., index]
            for name, index in self.output_index.items()
        }
        self.axis_units = dict(axis_units)
        self.output_units = dict(output_units)
        self.metadata = dict(metadata)
        self.hermite_outputs = self.metadata.get("hermite_outputs", {})
        if self.hermite_outputs:
            if axis_order != ("pressure", "temperature") or any(len(a) < 2 for a in self.axes.values()):
                raise ValueError("Hermite PT interpolation requires two pressure/temperature axes")
            for name, derivatives in self.hermite_outputs.items():
                if len(derivatives) != 3 or not {name, *derivatives} <= self.outputs.keys():
                    raise ValueError(f"Missing Hermite values/slopes for '{name}' in '{self.name}'")
        self.constants = dict(constants)
        self.status = dict(status)
        self.discrete_axes = frozenset(discrete_axes)
        self._warned_bounds = set()
        dimensions = len(self.axis_order)
        self._corner_bits = np.indices(
            (2,) * dimensions, dtype=np.intp
        ).reshape(dimensions, -1).T

    @classmethod
    def from_hdf5(
        cls,
        name: str,
        group: h5py.Group,
        discrete_axes: Iterable[str] = (),
    ) -> "LookupTable":
        axis_order = tuple(_json_attribute(group.attrs, "axis_order"))
        output_names = tuple(_json_attribute(group.attrs, "output_names"))
        metadata = _json_attribute(group.attrs, "metadata")
        constants = _json_attribute(group.attrs, "constants")
        if not axis_order or not output_names:
            raise ValueError(f"Lookup table '{name}' must have axes and outputs")
        if "axes" not in group or "outputs" not in group:
            raise ValueError(f"Lookup table '{name}' is missing axes or outputs")

        axes: Dict[str, np.ndarray] = {}
        axis_units: Dict[str, str] = {}
        for axis_name in axis_order:
            if axis_name not in group["axes"]:
                raise ValueError(f"Table '{name}' is missing axis '{axis_name}'")
            dataset = group["axes"][axis_name]
            values = np.asarray(dataset[...], dtype=float)
            if values.ndim != 1 or not len(values):
                raise ValueError(f"Axis '{axis_name}' in table '{name}' must be 1D")
            if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
                raise ValueError(
                    f"Axis '{axis_name}' in table '{name}' must be finite and increasing"
                )
            axes[axis_name] = values
            axis_units[axis_name] = str(dataset.attrs.get("units", ""))

        expected_shape = tuple(len(axes[axis]) for axis in axis_order)
        outputs: Dict[str, np.ndarray] = {}
        output_units: Dict[str, str] = {}
        for output_name in output_names:
            if output_name not in group["outputs"]:
                raise ValueError(f"Table '{name}' is missing output '{output_name}'")
            dataset = group["outputs"][output_name]
            values = np.asarray(dataset[...], dtype=float)
            if values.shape != expected_shape:
                raise ValueError(
                    f"Output '{output_name}' in table '{name}' has shape "
                    f"{values.shape}; expected {expected_shape}"
                )
            outputs[output_name] = values
            output_units[output_name] = str(dataset.attrs.get("units", ""))

        status: Dict[str, np.ndarray] = {}
        if "status" in group:
            for status_name, dataset in group["status"].items():
                values = np.asarray(dataset[...])
                if values.shape != expected_shape:
                    raise ValueError(
                        f"Status '{status_name}' in table '{name}' has shape "
                        f"{values.shape}; expected {expected_shape}"
                    )
                status[status_name] = values

        if not isinstance(metadata, dict) or not isinstance(constants, dict):
            raise ValueError(f"Table '{name}' metadata and constants must be objects")
        return cls(
            name,
            axis_order,
            axes,
            outputs,
            axis_units,
            output_units,
            metadata,
            constants,
            status,
            discrete_axes,
        )

    def _point(self, coordinates: Mapping[str, float]) -> Tuple[float, ...]:
        missing = set(self.axis_order).difference(coordinates)
        extra = set(coordinates).difference(self.axis_order)
        if missing or extra:
            raise ValueError(
                f"Invalid coordinates for '{self.name}': "
                f"missing={sorted(missing)}, unknown={sorted(extra)}"
            )
        point = tuple(float(coordinates[name]) for name in self.axis_order)
        if not np.all(np.isfinite(point)):
            raise ValueError(f"Coordinates for '{self.name}' must be finite")
        for name, value in zip(self.axis_order, point):
            if name in self.discrete_axes and not np.any(self.axes[name] == value):
                raise ValueError(
                    f"Axis '{name}' in table '{self.name}' requires one of "
                    f"{self.axes[name].tolist()}"
                )
        return point

    def _location(self, point: Tuple[float, ...]):
        lower = np.empty(len(point), dtype=np.intp)
        upper = np.empty(len(point), dtype=np.intp)
        fractions = np.empty(len(point), dtype=float)
        for dimension, (axis_name, coordinate) in enumerate(
            zip(self.axis_order, point)
        ):
            axis = self.axes[axis_name]
            if coordinate < axis[0] or coordinate > axis[-1]:
                key = (axis_name, "low" if coordinate < axis[0] else "high")
                if key not in self._warned_bounds:
                    print(
                        f"WARNING: rejected query outside table '{self.name}': "
                        f"{axis_name}={coordinate:.12g}, valid range="
                        f"[{axis[0]:.12g}, {axis[-1]:.12g}]. No extrapolation.",
                        file=sys.stderr,
                    )
                    self._warned_bounds.add(key)
                raise LookupBoundsError(
                    f"Coordinate '{axis_name}'={coordinate} is outside table "
                    f"'{self.name}' bounds [{axis[0]}, {axis[-1]}]",
                    table=self.name, axis=axis_name, value=float(coordinate),
                    lower=float(axis[0]), upper=float(axis[-1]),
                )
            upper_index = int(np.searchsorted(axis, coordinate))
            if upper_index < len(axis) and axis[upper_index] == coordinate:
                lower[dimension] = upper_index
                upper[dimension] = upper_index
                fractions[dimension] = 0.0
                continue
            lower_index = upper_index - 1
            lower[dimension] = lower_index
            upper[dimension] = upper_index
            fractions[dimension] = (
                (coordinate - axis[lower_index])
                / (axis[upper_index] - axis[lower_index])
            )
        return lower, upper, fractions

    @lru_cache(maxsize=128)
    def _columns(self, names: Tuple[str, ...]) -> Tuple[int, ...]:
        missing = set(names).difference(self.output_index)
        if missing:
            name = sorted(missing)[0]
            raise KeyError(f"Table '{self.name}' has no output '{name}'")
        return tuple(self.output_index[name] for name in names)

    @lru_cache(maxsize=8192)
    def _interpolate(
        self,
        point: Tuple[float, ...],
        columns: Tuple[int, ...],
    ) -> Tuple[float, ...]:
        """Interpolate selected packed outputs at one validated point."""

        lower, upper, fractions = self._location(point)
        indices = np.where(self._corner_bits, upper, lower)
        weights = np.prod(
            np.where(self._corner_bits, fractions, 1.0 - fractions), axis=1
        )
        active = weights > 0.0
        indices = indices[active]
        weights = weights[active]
        index = tuple(indices[:, dimension] for dimension in range(len(point)))
        coordinates = dict(zip(self.axis_order, point))

        if "success" in self.status and not np.all(self.status["success"][index]):
            raise ValueError(
                f"Table '{self.name}' contains an unsuccessful state at "
                f"{coordinates}"
            )

        corner_values = np.take(self._values[index], columns, axis=-1)
        if not np.all(np.isfinite(corner_values)):
            invalid_column = int(np.argwhere(~np.isfinite(corner_values))[0, 1])
            invalid = self.output_names[columns[invalid_column]]
            raise ValueError(
                f"Table '{self.name}' returned a non-finite '{invalid}' at "
                f"{coordinates}"
            )
        values = weights @ corner_values
        if self.hermite_outputs:
            smooth = self._hermite(point)
            values = [smooth.get(self.output_names[column], value)
                      for column, value in zip(columns, values)]
        return tuple(float(value) for value in values)

    @staticmethod
    def _hermite_basis(s, width):
        return (
            np.array([2*s**3 - 3*s*s + 1, -2*s**3 + 3*s*s, s**3 - 2*s*s + s, s**3 - s*s]),
            np.array([6*s*s - 6*s, -6*s*s + 6*s, 3*s*s - 4*s + 1, 3*s*s - 2*s]) / width,
        )

    @lru_cache(maxsize=8192)
    def _hermite(self, point):
        """Bicubic PT values/slopes from the same stored corner data.

        Never bridge invalid cells or fall back to a discontinuous interpolant.
        At a knot shared with an invalid cell, a valid neighboring cell suffices.
        """
        self._location(point)  # Bounds and no-extrapolation policy apply here too.
        candidates = []
        for axis_name, coordinate in zip(self.axis_order, point):
            axis = self.axes[axis_name]
            i = int(np.clip(np.searchsorted(axis, coordinate, side="right") - 1, 0, len(axis) - 2))
            candidates.append([i, i-1] if i > 0 and coordinate == axis[i] else [i])
        cell = None
        for i in candidates[0]:
            for j in candidates[1]:
                sl = np.s_[i:i+2, j:j+2]
                if "success" not in self.status or np.all(self.status["success"][sl]):
                    cell = (i, j)
                    break
            if cell is not None:
                break
        if cell is None:
            raise ValueError(f"Table '{self.name}' contains an unsuccessful state in the PT interpolation cell at {point}")
        i, j = cell
        pressure, temperature = (self.axes[k] for k in self.axis_order)
        dp, dt = pressure[i+1] - pressure[i], temperature[j+1] - temperature[j]
        bp, dbp = self._hermite_basis((point[0] - pressure[i]) / dp, dp)
        bt, dbt = self._hermite_basis((point[1] - temperature[j]) / dt, dt)
        result = {}
        for name, (dP, dT, dPT) in self.hermite_outputs.items():
            g = np.empty((4, 4))
            g[:2, :2] = self.outputs[name][sl]
            g[2:, :2] = self.outputs[dP][sl] * dp
            g[:2, 2:] = self.outputs[dT][sl] * dt
            g[2:, 2:] = self.outputs[dPT][sl] * dp * dt
            if not np.isfinite(g).all():
                raise ValueError(f"Table '{self.name}' has non-finite Hermite data for '{name}' at {point}")
            result.update({name: float(bp @ g @ bt), dP: float(dbp @ g @ bt),
                           dT: float(bp @ g @ dbt), dPT: float(dbp @ g @ dbt)})
        if {"density", "internal_energy"} <= result.keys():
            if result["density"] <= 0 or not np.isfinite(list(result.values())).all():
                raise ValueError(f"Table '{self.name}' returned invalid Hermite properties at {point}")
            result["enthalpy"] = result["internal_energy"] + point[0] / result["density"]
        return result

    def get(self, output_name: str, **coordinates: float) -> float:
        """Interpolate one output without extrapolation."""

        return self.evaluate((output_name,), **coordinates)[output_name]

    def derivative(self, output_names: Iterable[str], *, wrt: str,
                   **coordinates: float) -> Dict[str, float]:
        """Differentiate the actual interpolant along one axis.

        PT Hermite outputs use analytic first derivatives. Other outputs retain
        the exact multilinear cell derivative, with the right-hand cell selected
        at a grid knot except at the upper boundary.
        """
        point = self._point(coordinates)
        self._location(point)
        if wrt not in self.axes or wrt in self.discrete_axes:
            raise ValueError(f"Cannot differentiate table '{self.name}' along {wrt!r}")
        axis = self.axes[wrt]
        if len(axis) < 2:
            raise ValueError(f"Derivative along '{wrt}' requires at least two grid points")
        i = int(np.clip(np.searchsorted(axis, coordinates[wrt], side="right") - 1,
                        0, len(axis) - 2))
        names = tuple(output_names)
        self._columns(names)
        if self.hermite_outputs:
            smooth = self._hermite(point)
            result = {}
            for name in names:
                if any(name in columns for columns in self.hermite_outputs.values()):
                    raise ValueError("Differentiate the PT property itself; derivatives of stored slope columns are not supported")
                if name in self.hermite_outputs:
                    column = self.hermite_outputs[name][0 if wrt == "pressure" else 1]
                    result[name] = smooth[column]
                elif name == "enthalpy" and {"density", "internal_energy"} <= self.hermite_outputs.keys():
                    k = 0 if wrt == "pressure" else 1
                    rho, u = self.hermite_outputs["density"], self.hermite_outputs["internal_energy"]
                    result[name] = (smooth[u[k]] - coordinates["pressure"] * smooth[rho[k]] / smooth["density"]**2
                                    + (1 / smooth["density"] if wrt == "pressure" else 0.))
            if len(result) == len(names):
                return result
        else:
            result = {}
        low = self.evaluate(names, **{**coordinates, wrt: float(axis[i])})
        high = self.evaluate(names, **{**coordinates, wrt: float(axis[i + 1])})
        return {name: result.get(name, (high[name] - low[name]) / float(axis[i + 1] - axis[i]))
                for name in names}

    def evaluate(
        self,
        output_names: Optional[Iterable[str]] = None,
        **coordinates: float,
    ) -> Dict[str, float]:
        names = self.output_names if output_names is None else tuple(output_names)
        point = self._point(coordinates)
        columns = self._columns(names)
        return dict(zip(names, self._interpolate(point, columns)))


class LookupTables:
    """Collection of selected rectangular-grid tables from one HDF5 file."""

    def __init__(
        self,
        path: str | Path,
        table_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Lookup file does not exist: {self.path}")

        requested = None if table_names is None else tuple(table_names)
        self.tables: Dict[str, LookupTable] = {}
        with h5py.File(self.path, "r") as file:
            self.attributes = dict(file.attrs)
            names = tuple(file) if requested is None else requested
            for name in names:
                if name not in file or not isinstance(file[name], h5py.Group):
                    raise KeyError(f"No lookup table named '{name}'")
                discrete_axes = ("nfz",) if name == "engine_lookup" else ()
                self.tables[name] = LookupTable.from_hdf5(
                    name, file[name], discrete_axes
                )
        if not self.tables:
            raise ValueError(f"Lookup file contains no tables: {self.path}")

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self.tables)

    def __getitem__(self, table_name: str) -> LookupTable:
        try:
            return self.tables[table_name]
        except KeyError as error:
            raise KeyError(f"No loaded lookup table named '{table_name}'") from error

    def get(self, table_name: str, output_name: str, **coordinates: float) -> float:
        return self[table_name].get(output_name, **coordinates)

    def evaluate(
        self,
        table_name: str,
        output_names: Optional[Iterable[str]] = None,
        **coordinates: float,
    ) -> Dict[str, float]:
        return self[table_name].evaluate(output_names, **coordinates)
