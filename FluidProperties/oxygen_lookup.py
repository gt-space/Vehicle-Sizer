from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

from datetime import datetime, timezone
import json

import h5py
import numpy as np

from thermoprop import Fluid


# ---------------------------------------------------------------------------
# File and HDF5 group
# ---------------------------------------------------------------------------

# Only /oxygen_hp is replaced. Existing groups in sizer_lookups.h5 are kept.
filename = "sizer_lookups"
group = "oxygen_lookup"

psia_to_pa = 6894.757293168361


# ---------------------------------------------------------------------------
# Map range and resolution
# ---------------------------------------------------------------------------

# Both lookup inputs use SI units and linear axes.
minimum_pressure = 101325.0
maximum_pressure = 700.0 * psia_to_pa

# The enthalpy axis is selected so that the map covers oxygen states from
# deeply cryogenic liquid through saturation/two-phase states and into hot gas.
minimum_temperature = 55.0
maximum_temperature = 1000.0

# 100 * 400 = 40,000 HP states. Five float64 outputs require only about
# 1.53 MiB before HDF5 overhead, so this stays comfortably in the MB range.
pressure_count = 100
enthalpy_count = 400

# Two workers provide multiprocessing without loading every CPU core. Increase
# this to 4 if maximum generation speed matters more than fan noise.
workers = min(2, os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

# These are the only datasets written under /oxygen_hp/outputs.
output_names = [
    "temperature",
    "density",
    "dynamic_viscosity",
    "conductivity",
    "specific_heat",
]

output_units = {
    "temperature": "K",
    "density": "kg/m^3",
    "dynamic_viscosity": "Pa*s",
    "conductivity": "W/(m*K)",
    "specific_heat": "J/(kg*K)",
}

metadata = {
    "description": "Oxygen properties tabulated by pressure and enthalpy",
    "fluid": "Oxygen",
    "state_inputs": ["pressure", "enthalpy"],
    "outputs": output_names,
    "pressure_units": "Pa",
    "enthalpy_units": "J/kg",
    "enthalpy_basis": "ThermoProp Fluid / CoolProp",
    "temperature_range_K": [minimum_temperature, maximum_temperature],
    "pressure_range_Pa": [minimum_pressure, maximum_pressure],
    "phase_region": "liquid, two-phase, gas, and near-critical states",
    "invalid_state_behavior": "generation stops before writing if any output is non-finite",
    "private_thermoprop_api_used": False,
}


# ---------------------------------------------------------------------------
# Worker-local state
# ---------------------------------------------------------------------------

_worker_oxygen: Fluid | None = None
_worker_enthalpy_axis: np.ndarray | None = None


def finite_float(value, name: str) -> float:
    """Return a finite float or raise a clear error."""

    if value is None:
        raise ValueError(f"{name} is unavailable.")

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} is not finite.")

    return value


def initialize_worker(enthalpy_axis: np.ndarray) -> None:
    """Create one reusable public Fluid object inside each worker process."""

    global _worker_oxygen
    global _worker_enthalpy_axis

    _worker_enthalpy_axis = np.asarray(enthalpy_axis, dtype=float)
    _worker_oxygen = Fluid(
        "Oxygen",
        pressure=minimum_pressure,
        enthalpy=float(_worker_enthalpy_axis[0]),
    )


def empty_row() -> dict[str, np.ndarray]:
    """Return one NaN-filled output row."""

    if _worker_enthalpy_axis is None:
        raise RuntimeError("The worker enthalpy axis has not been initialized.")

    return {
        name: np.full(_worker_enthalpy_axis.size, np.nan, dtype=float)
        for name in output_names
    }


def evaluate_pressure_row(job: tuple[int, float]):
    """Evaluate one complete pressure row using public HP flashes.

    Fluid naturally supports pressure-enthalpy flashes, including single-phase
    liquid, two-phase, gas, and near-critical states. The same Fluid object is
    reused for every enthalpy value in the row through the public
    ``pressure_enthalpy`` setter.
    """

    if _worker_oxygen is None or _worker_enthalpy_axis is None:
        raise RuntimeError("The worker oxygen model has not been initialized.")

    row_index, pressure = job
    pressure = float(pressure)
    row = empty_row()
    valid_points = 0
    first_error: str | None = None

    for column_index, enthalpy in enumerate(_worker_enthalpy_axis):
        try:
            _worker_oxygen.pressure_enthalpy = (
                pressure,
                float(enthalpy),
            )

            temperature = finite_float(
                _worker_oxygen.temperature,
                "oxygen temperature",
            )

            # The common enthalpy axis is the intersection of the valid
            # 55-1000 K enthalpy intervals at every pressure, so every flashed
            # temperature should remain inside the requested range.
            if not (minimum_temperature <= temperature <= maximum_temperature):
                raise ValueError(
                    f"Flashed temperature {temperature:.6g} K is outside "
                    f"{minimum_temperature:.6g}-{maximum_temperature:.6g} K."
                )

            values = {
                "temperature": temperature,
                "density": finite_float(
                    _worker_oxygen.density,
                    "oxygen density",
                ),
                "dynamic_viscosity": finite_float(
                    _worker_oxygen.dynamic_viscosity,
                    "oxygen dynamic viscosity",
                ),
                "conductivity": finite_float(
                    _worker_oxygen.conductivity,
                    "oxygen conductivity",
                ),
                "specific_heat": finite_float(
                    _worker_oxygen.specific_heat,
                    "oxygen specific heat",
                ),
            }

            for name, value in values.items():
                row[name][column_index] = value

            valid_points += 1

        except Exception as exc:
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"

    return row_index, row, valid_points, first_error


# ---------------------------------------------------------------------------
# Axis construction
# ---------------------------------------------------------------------------


def enthalpy_axis_limits(pressure_axis: np.ndarray) -> tuple[float, float]:
    """Find common enthalpy limits using only public PT states.

    Each pressure has a slightly different enthalpy interval between 55 K and
    1000 K. The common map axis uses the intersection of all those intervals:

        h_min = max[h(P, 55 K)]
        h_max = min[h(P, 1000 K)]

    This keeps every pressure-enthalpy combination valid and finite, which is
    required by FullFlow's rectangular ``Map`` component.
    """

    oxygen = Fluid(
        "Oxygen",
        pressure=float(pressure_axis[0]),
        temperature=minimum_temperature,
    )

    low_enthalpies = np.empty(pressure_axis.size, dtype=float)
    high_enthalpies = np.empty(pressure_axis.size, dtype=float)

    for index, pressure in enumerate(pressure_axis):
        oxygen.pressure_temperature = (
            float(pressure),
            minimum_temperature,
        )
        low_enthalpies[index] = finite_float(
            oxygen.enthalpy,
            "oxygen minimum-temperature enthalpy",
        )

        oxygen.pressure_temperature = (
            float(pressure),
            maximum_temperature,
        )
        high_enthalpies[index] = finite_float(
            oxygen.enthalpy,
            "oxygen maximum-temperature enthalpy",
        )

    minimum_common_enthalpy = float(np.max(low_enthalpies))
    maximum_common_enthalpy = float(np.min(high_enthalpies))

    if minimum_common_enthalpy >= maximum_common_enthalpy:
        raise ValueError(
            "The requested oxygen pressure and temperature ranges do not "
            "share a common enthalpy interval."
        )

    return minimum_common_enthalpy, maximum_common_enthalpy


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------


def precompute_map(
    pressure_axis: np.ndarray,
    enthalpy_axis: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate every pressure-enthalpy state with multiprocessing."""

    shape = (pressure_axis.size, enthalpy_axis.size)
    total_points = int(np.prod(shape))

    values = {
        name: np.full(shape, np.nan, dtype=float)
        for name in output_names
    }

    jobs = [
        (index, float(pressure))
        for index, pressure in enumerate(pressure_axis)
    ]

    print()
    print(
        f"{group}: evaluating {pressure_axis.size:,} pressure rows and "
        f"{total_points:,} HP points with {workers} worker(s)"
    )
    print(
        f"{group}: direct public Fluid pressure-enthalpy flashes; "
        "no root solve and no private ThermoProp methods"
    )
    print()

    start_time = time.perf_counter()
    completed_rows = 0
    valid_points = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(enthalpy_axis,),
    ) as executor:
        futures = [
            executor.submit(evaluate_pressure_row, job)
            for job in jobs
        ]

        for future in as_completed(futures):
            row_index, row, row_valid_points, first_error = future.result()

            for name in output_names:
                values[name][row_index, :] = row[name]

            completed_rows += 1
            valid_points += row_valid_points

            elapsed = time.perf_counter() - start_time
            row_rate = completed_rows / elapsed if elapsed > 0.0 else 0.0
            remaining_rows = pressure_axis.size - completed_rows
            eta = remaining_rows / row_rate if row_rate > 0.0 else np.nan

            message = (
                f"{group}: row {completed_rows:,}/{pressure_axis.size:,} | "
                f"valid={valid_points:,}/{total_points:,} | "
                f"elapsed={elapsed / 60.0:.1f} min | "
                f"eta={eta / 60.0:.1f} min"
            )

            # A few individual invalid edge flashes are expected. This only
            # reports the first exception from a row when one occurred.
            if first_error is not None and row_valid_points == 0:
                message += (
                    f" | pressure={pressure_axis[row_index]:.6g} Pa"
                    f" | {first_error}"
                )

            print(message, flush=True)

    nonfinite = {
        name: int(np.count_nonzero(~np.isfinite(array)))
        for name, array in values.items()
    }
    nonfinite = {name: count for name, count in nonfinite.items() if count}

    if nonfinite:
        raise RuntimeError(
            f"{group}: generated non-finite output values: {nonfinite}. "
            "The map was not written."
        )

    print()
    print(
        f"{group}: property evaluation complete; "
        f"valid states={valid_points:,}/{total_points:,}"
    )

    return values


# ---------------------------------------------------------------------------
# HDF5 map writing
# ---------------------------------------------------------------------------


def h5_filename(path: str) -> str:
    """Return the HDF5 filename used by the lookup file."""

    path = str(path)
    return path if path.endswith(".h5") else f"{path}.h5"


def write_map(
    pressure_axis: np.ndarray,
    enthalpy_axis: np.ndarray,
    values: dict[str, np.ndarray],
) -> str:
    """Write a FullFlow-compatible rectangular HDF5 map directly.

    FullFlow's ``Map.from_hdf5`` requires only ``axes``, ``outputs``, and the
    ``axis_order`` attribute. Writing the already-computed arrays directly is
    much faster than calling a point-by-point map-generation callback.
    """

    path = h5_filename(filename)

    with h5py.File(path, "a") as file:
        if group in file:
            del file[group]

        map_group = file.create_group(group)
        map_group.attrs["kind"] = "map"
        map_group.attrs["name"] = group
        map_group.attrs["map_format"] = "rectangular-grid-map-v1"
        map_group.attrs["axis_order"] = json.dumps(["pressure", "enthalpy"])
        map_group.attrs["output_names"] = json.dumps(output_names)
        map_group.attrs["constants"] = json.dumps({})
        map_group.attrs["metadata"] = json.dumps(metadata)
        map_group.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

        axes_group = map_group.create_group("axes")

        pressure_dataset = axes_group.create_dataset(
            "pressure",
            data=np.asarray(pressure_axis, dtype=float),
        )
        pressure_dataset.attrs["name"] = "pressure"
        pressure_dataset.attrs["spacing"] = "linear"
        pressure_dataset.attrs["units"] = "Pa"

        enthalpy_dataset = axes_group.create_dataset(
            "enthalpy",
            data=np.asarray(enthalpy_axis, dtype=float),
        )
        enthalpy_dataset.attrs["name"] = "enthalpy"
        enthalpy_dataset.attrs["spacing"] = "linear"
        enthalpy_dataset.attrs["units"] = "J/kg"

        outputs_group = map_group.create_group("outputs")

        for name in output_names:
            dataset = outputs_group.create_dataset(
                name,
                data=np.asarray(values[name], dtype=float),
            )
            dataset.attrs["units"] = output_units[name]

        # Status is optional for FullFlow, but keeping it makes the file easy to
        # inspect with the same tools used for FullPlot-generated maps.
        status_group = map_group.create_group("status")
        shape = (pressure_axis.size, enthalpy_axis.size)
        status_group.create_dataset(
            "success",
            data=np.ones(shape, dtype=bool),
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        status_group.create_dataset(
            "message",
            data=np.full(shape, "", dtype=object),
            dtype=string_dtype,
        )

    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    pressure_axis = np.linspace(
        minimum_pressure,
        maximum_pressure,
        pressure_count,
        dtype=float,
    )

    minimum_enthalpy, maximum_enthalpy = enthalpy_axis_limits(pressure_axis)

    enthalpy_axis = np.linspace(
        minimum_enthalpy,
        maximum_enthalpy,
        enthalpy_count,
        dtype=float,
    )

    total_points = pressure_count * enthalpy_count
    raw_output_size_mib = (
        total_points * len(output_names) * 8.0 / (1024.0**2)
    )

    print()
    print(f"{group}: target file = {filename}.h5")
    print(
        f"{group}: pressure range = {minimum_pressure:.6g} to "
        f"{maximum_pressure:.6g} Pa "
        f"({minimum_pressure / psia_to_pa:.2f} to "
        f"{maximum_pressure / psia_to_pa:.1f} psia)"
    )
    print(
        f"{group}: temperature coverage = "
        f"{minimum_temperature:.1f} to {maximum_temperature:.1f} K"
    )
    print(
        f"{group}: enthalpy range = {minimum_enthalpy:.6g} to "
        f"{maximum_enthalpy:.6g} J/kg"
    )
    print(f"{group}: map shape = ({pressure_count}, {enthalpy_count})")
    print(f"{group}: total HP states = {total_points:,}")
    print(
        f"{group}: raw five-output payload = "
        f"approximately {raw_output_size_mib:.2f} MiB"
    )
    print()

    values = precompute_map(
        pressure_axis=pressure_axis,
        enthalpy_axis=enthalpy_axis,
    )

    output_path = write_map(
        pressure_axis=pressure_axis,
        enthalpy_axis=enthalpy_axis,
        values=values,
    )

    print()
    print(f"{group}: done")
    print(f"{group}: wrote {output_path}")
    print()


if __name__ == "__main__":
    main()
