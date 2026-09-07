from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import h5py
import numpy as np


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
        self.outputs = dict(outputs)
        self.axis_units = dict(axis_units)
        self.output_units = dict(output_units)
        self.metadata = dict(metadata)
        self.constants = dict(constants)
        self.status = dict(status)
        self.discrete_axes = frozenset(discrete_axes)

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

    def get(self, output_name: str, **coordinates: float) -> float:
        """Linearly interpolate one output without extrapolation."""

        if output_name not in self.outputs:
            raise KeyError(f"Table '{self.name}' has no output '{output_name}'")
        point = self._point(coordinates)
        brackets = []
        for axis_name, coordinate in zip(self.axis_order, point):
            axis = self.axes[axis_name]
            if coordinate < axis[0] or coordinate > axis[-1]:
                raise ValueError(
                    f"Coordinates are outside table '{self.name}': {coordinates}"
                )
            exact = np.flatnonzero(axis == coordinate)
            if len(exact):
                brackets.append(((int(exact[0]), 1.0),))
                continue
            upper = int(np.searchsorted(axis, coordinate))
            lower = upper - 1
            fraction = (coordinate - axis[lower]) / (axis[upper] - axis[lower])
            brackets.append(((lower, 1.0 - fraction), (upper, fraction)))

        value = 0.0
        for corner in product(*brackets):
            index = tuple(item[0] for item in corner)
            weight = float(np.prod([item[1] for item in corner]))
            if "success" in self.status and not bool(self.status["success"][index]):
                raise ValueError(
                    f"Table '{self.name}' contains an unsuccessful state at "
                    f"{coordinates}"
                )
            corner_value = self.outputs[output_name][index]
            if not np.isfinite(corner_value):
                raise ValueError(
                    f"Table '{self.name}' returned a non-finite '{output_name}' at "
                    f"{coordinates}"
                )
            value += weight * float(corner_value)
        return value

    def evaluate(
        self,
        output_names: Optional[Iterable[str]] = None,
        **coordinates: float,
    ) -> Dict[str, float]:
        names = tuple(self.outputs) if output_names is None else tuple(output_names)
        return {name: self.get(name, **coordinates) for name in names}


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
