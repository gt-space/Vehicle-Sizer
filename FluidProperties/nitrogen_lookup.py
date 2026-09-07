from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import time

import h5py
import numpy as np

from thermoprop import Fluid


# ---------------------------------------------------------------------------
# File and HDF5 groups
# ---------------------------------------------------------------------------

# Both gas maps are generated together. Existing unrelated groups are preserved;
# only /nitrogen_hp and /helium_hp are replaced after both maps validate.
filename = "sizer_lookups.h5"

psia_to_pa = 6894.757293168361


@dataclass(frozen=True)
class GasSpecification:
    fluid_name: str
    group_name: str


gas_specifications = (
    GasSpecification(fluid_name="Nitrogen", group_name="nitrogen_hp"),
    GasSpecification(fluid_name="Helium", group_name="helium_hp"),
)


# ---------------------------------------------------------------------------
# Map range and resolution
# ---------------------------------------------------------------------------

# Both lookup inputs are SI and linearly spaced. The pressure range extends well
# above 6,000 psia for high-pressure COPV storage and blowdown calculations.
minimum_pressure = 101325.0
maximum_pressure = 10_000.0 * psia_to_pa

# Both fluids remain single-phase above 150 K over the requested pressure range.
# States may be reported by the backend as gas, supercritical gas, or
# supercritical fluid; liquid and two-phase states are not accepted.
minimum_temperature = 150.0
maximum_temperature = 1000.0

# Each fluid contains 200 * 400 = 80,000 pressure-enthalpy states.
pressure_count = 200
enthalpy_count = 400

# Two workers keep the machine responsive while still using multiprocessing.
workers = min(2, os.cpu_count() or 1)

# Avoid placing the first and last enthalpy samples exactly on PT-derived
# boundaries, where backend floating-point roundoff can reject a valid endpoint.
enthalpy_endpoint_margin_fraction = 1.0e-7

allowed_phase_names = {
    "Gas",
    "SupercriticalGas",
    "Supercritical",
}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output_names = [
    "temperature",
    "density",
    "dynamic_viscosity",
    "conductivity",
    "specific_heat_at_constant_pressure",
    "specific_heat_at_constant_volume",
    "specific_heat_ratio",
]

output_units = {
    "temperature": "K",
    "density": "kg/m^3",
    "dynamic_viscosity": "Pa*s",
    "conductivity": "W/(m*K)",
    "specific_heat_at_constant_pressure": "J/(kg*K)",
    "specific_heat_at_constant_volume": "J/(kg*K)",
    "specific_heat_ratio": "dimensionless",
}


# ---------------------------------------------------------------------------
# Worker-local state
# ---------------------------------------------------------------------------

_worker_fluid: Fluid | None = None
_worker_enthalpy_axis: np.ndarray | None = None
_worker_specification: GasSpecification | None = None


def finite_float(value, name: str) -> float:
    """Return a finite float or raise a clear error."""

    if value is None:
        raise ValueError(f"{name} is unavailable.")

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} is not finite.")

    return value


def positive_finite_float(value, name: str) -> float:
    """Return a positive finite float or raise a clear error."""

    value = finite_float(value, name)

    if value <= 0.0:
        raise ValueError(f"{name} must be positive; received {value:.12g}.")

    return value


def initialize_worker(
    specification: GasSpecification,
    enthalpy_axis: np.ndarray,
) -> None:
    """Create one reusable public Fluid object inside each worker process."""

    global _worker_fluid
    global _worker_enthalpy_axis
    global _worker_specification

    _worker_specification = specification
    _worker_enthalpy_axis = np.asarray(enthalpy_axis, dtype=float)
    _worker_fluid = Fluid(
        specification.fluid_name,
        pressure=minimum_pressure,
        enthalpy=float(_worker_enthalpy_axis[0]),
    )


def empty_row() -> dict[str, np.ndarray]:
    """Return one NaN-filled row for every requested output."""

    if _worker_enthalpy_axis is None:
        raise RuntimeError("The worker enthalpy axis has not been initialized.")

    return {
        name: np.full(_worker_enthalpy_axis.size, np.nan, dtype=float)
        for name in output_names
    }


def evaluate_pressure_row(job: tuple[int, float]):
    """Evaluate one complete pressure row with public Fluid HP flashes."""

    if (
        _worker_fluid is None
        or _worker_enthalpy_axis is None
        or _worker_specification is None
    ):
        raise RuntimeError("The worker fluid model has not been initialized.")

    row_index, pressure = job
    pressure = float(pressure)
    row = empty_row()
    valid_points = 0
    errors: list[str] = []
    phase_counts: dict[str, int] = {}
    fluid_name = _worker_specification.fluid_name

    for column_index, enthalpy in enumerate(_worker_enthalpy_axis):
        try:
            _worker_fluid.pressure_enthalpy = (
                pressure,
                float(enthalpy),
            )

            temperature = finite_float(
                _worker_fluid.temperature,
                f"{fluid_name} temperature",
            )

            if not (minimum_temperature <= temperature <= maximum_temperature):
                raise ValueError(
                    f"{fluid_name} flashed temperature {temperature:.12g} K is "
                    f"outside {minimum_temperature:.12g}-{maximum_temperature:.12g} K."
                )

            phase_name = str(_worker_fluid.phase)
            if phase_name not in allowed_phase_names:
                raise ValueError(
                    f"{fluid_name} phase {phase_name!r} is not a supported "
                    "single-phase gas or supercritical state."
                )

            constant_pressure_specific_heat = positive_finite_float(
                _worker_fluid.specific_heat_cp,
                f"{fluid_name} specific heat at constant pressure",
            )
            constant_volume_specific_heat = positive_finite_float(
                _worker_fluid.specific_heat_cv,
                f"{fluid_name} specific heat at constant volume",
            )
            specific_heat_ratio = positive_finite_float(
                _worker_fluid.specific_heat_ratio,
                f"{fluid_name} specific heat ratio",
            )

            calculated_specific_heat_ratio = (
                constant_pressure_specific_heat / constant_volume_specific_heat
            )
            if not np.isclose(
                specific_heat_ratio,
                calculated_specific_heat_ratio,
                rtol=1.0e-12,
                atol=0.0,
            ):
                raise ValueError(
                    f"{fluid_name} specific heat ratio {specific_heat_ratio:.12g} "
                    "does not equal the constant-pressure to constant-volume "
                    f"specific heat ratio {calculated_specific_heat_ratio:.12g}."
                )

            values = {
                "temperature": temperature,
                "density": positive_finite_float(
                    _worker_fluid.density,
                    f"{fluid_name} density",
                ),
                "dynamic_viscosity": positive_finite_float(
                    _worker_fluid.dynamic_viscosity,
                    f"{fluid_name} dynamic viscosity",
                ),
                "conductivity": positive_finite_float(
                    _worker_fluid.conductivity,
                    f"{fluid_name} conductivity",
                ),
                "specific_heat_at_constant_pressure": constant_pressure_specific_heat,
                "specific_heat_at_constant_volume": constant_volume_specific_heat,
                "specific_heat_ratio": specific_heat_ratio,
            }

            for name, value in values.items():
                row[name][column_index] = value

            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
            valid_points += 1

        except Exception as exc:
            if len(errors) < 3:
                errors.append(
                    f"column={column_index}, h={float(enthalpy):.12g} J/kg, "
                    f"{type(exc).__name__}: {exc}"
                )

    return row_index, row, valid_points, errors, phase_counts


# ---------------------------------------------------------------------------
# Axis construction
# ---------------------------------------------------------------------------


def enthalpy_axis_limits(
    specification: GasSpecification,
    pressure_axis: np.ndarray,
) -> tuple[float, float]:
    """Find a common 150-1000 K enthalpy interval for one fluid.

    At each pressure, public pressure-temperature states define the enthalpy
    interval associated with the nominal temperature range. Intersecting those
    intervals gives one rectangular pressure-enthalpy table whose flashes stay
    inside the requested single-phase region at every pressure.
    """

    fluid = Fluid(
        specification.fluid_name,
        pressure=float(pressure_axis[0]),
        temperature=minimum_temperature,
    )

    low_enthalpies = np.empty(pressure_axis.size, dtype=float)
    high_enthalpies = np.empty(pressure_axis.size, dtype=float)

    for index, pressure in enumerate(pressure_axis):
        fluid.pressure_temperature = (
            float(pressure),
            minimum_temperature,
        )
        low_enthalpies[index] = finite_float(
            fluid.enthalpy,
            f"{specification.fluid_name} minimum-temperature enthalpy",
        )

        fluid.pressure_temperature = (
            float(pressure),
            maximum_temperature,
        )
        high_enthalpies[index] = finite_float(
            fluid.enthalpy,
            f"{specification.fluid_name} maximum-temperature enthalpy",
        )

    minimum_common_enthalpy = float(np.max(low_enthalpies))
    maximum_common_enthalpy = float(np.min(high_enthalpies))

    if minimum_common_enthalpy >= maximum_common_enthalpy:
        raise ValueError(
            f"The requested {specification.fluid_name} pressure and temperature "
            "ranges do not share a common enthalpy interval."
        )

    span = maximum_common_enthalpy - minimum_common_enthalpy
    margin = max(1.0e-6, enthalpy_endpoint_margin_fraction * span)

    minimum_common_enthalpy += margin
    maximum_common_enthalpy -= margin

    if minimum_common_enthalpy >= maximum_common_enthalpy:
        raise ValueError("The enthalpy endpoint margin removed the usable interval.")

    return minimum_common_enthalpy, maximum_common_enthalpy


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------


def precompute_map(
    specification: GasSpecification,
    pressure_axis: np.ndarray,
    enthalpy_axis: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Evaluate and validate every pressure-enthalpy state for one fluid."""

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
        f"{specification.group_name}: evaluating "
        f"{pressure_axis.size:,} pressure rows and {total_points:,} HP points "
        f"with {workers} worker(s)"
    )
    print(
        f"{specification.group_name}: public ThermoProp Fluid HP flashes; "
        "only gas and supercritical states are accepted"
    )
    print()

    start_time = time.perf_counter()
    completed_rows = 0
    valid_points = 0
    failed_examples: list[str] = []
    phase_counts: dict[str, int] = {}

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(specification, enthalpy_axis),
    ) as executor:
        futures = [
            executor.submit(evaluate_pressure_row, job)
            for job in jobs
        ]

        for future in as_completed(futures):
            (
                row_index,
                row,
                row_valid_points,
                row_errors,
                row_phase_counts,
            ) = future.result()

            for name in output_names:
                values[name][row_index, :] = row[name]

            completed_rows += 1
            valid_points += row_valid_points

            for phase_name, count in row_phase_counts.items():
                phase_counts[phase_name] = phase_counts.get(phase_name, 0) + count

            for error in row_errors:
                if len(failed_examples) < 10:
                    failed_examples.append(
                        f"pressure={pressure_axis[row_index]:.12g} Pa, {error}"
                    )

            elapsed = time.perf_counter() - start_time
            row_rate = completed_rows / elapsed if elapsed > 0.0 else 0.0
            remaining_rows = pressure_axis.size - completed_rows
            eta = remaining_rows / row_rate if row_rate > 0.0 else np.nan

            print(
                f"{specification.group_name}: row "
                f"{completed_rows:,}/{pressure_axis.size:,} | "
                f"valid={valid_points:,}/{total_points:,} | "
                f"elapsed={elapsed / 60.0:.1f} min | "
                f"eta={eta / 60.0:.1f} min",
                flush=True,
            )

    nonfinite = {
        name: int(np.count_nonzero(~np.isfinite(array)))
        for name, array in values.items()
    }
    nonfinite = {name: count for name, count in nonfinite.items() if count}

    if nonfinite or valid_points != total_points:
        details = "\n".join(f"  - {example}" for example in failed_examples)
        if details:
            details = f"\nExample failures:\n{details}"

        raise RuntimeError(
            f"{specification.group_name}: property evaluation failed; "
            f"valid={valid_points:,}/{total_points:,}, "
            f"non-finite outputs={nonfinite}. No HDF5 groups were replaced."
            f"{details}"
        )

    print()
    print(
        f"{specification.group_name}: property evaluation complete; "
        f"valid states={valid_points:,}/{total_points:,}; "
        f"phases={dict(sorted(phase_counts.items()))}"
    )

    return values, phase_counts


# ---------------------------------------------------------------------------
# HDF5 map writing
# ---------------------------------------------------------------------------


@dataclass
class GeneratedGasMap:
    specification: GasSpecification
    pressure_axis: np.ndarray
    enthalpy_axis: np.ndarray
    values: dict[str, np.ndarray]
    phase_counts: dict[str, int]


def map_metadata(generated_map: GeneratedGasMap) -> dict[str, object]:
    """Return complete, JSON-serializable metadata for one gas map."""

    specification = generated_map.specification

    return {
        "description": (
            f"{specification.fluid_name} single-phase gas and supercritical-fluid "
            "properties tabulated by pressure and enthalpy for COPV analysis"
        ),
        "fluid": specification.fluid_name,
        "state_inputs": ["pressure", "enthalpy"],
        "outputs": output_names,
        "pressure_units": "Pa",
        "pressure_range_Pa": [minimum_pressure, maximum_pressure],
        "pressure_range_psia": [
            minimum_pressure / psia_to_pa,
            maximum_pressure / psia_to_pa,
        ],
        "pressure_spacing": "linear",
        "pressure_count": pressure_count,
        "enthalpy_units": "J/kg",
        "enthalpy_range_J_per_kg": [
            float(generated_map.enthalpy_axis[0]),
            float(generated_map.enthalpy_axis[-1]),
        ],
        "enthalpy_spacing": "linear",
        "enthalpy_count": enthalpy_count,
        "enthalpy_basis": "ThermoProp Fluid public API / CoolProp backend",
        "nominal_temperature_range_K": [
            minimum_temperature,
            maximum_temperature,
        ],
        "phase_region": "single-phase gas and supercritical fluid",
        "accepted_phase_labels": sorted(allowed_phase_names),
        "observed_phase_point_counts": dict(sorted(generated_map.phase_counts.items())),
        "thermophysical_property_source": "ThermoProp Fluid public API",
        "specific_heat_at_constant_pressure_source": "Fluid.specific_heat_cp",
        "specific_heat_at_constant_volume_source": "Fluid.specific_heat_cv",
        "specific_heat_ratio_source": "Fluid.specific_heat_ratio (Cp/Cv)",
        "property_validation": (
            "all output values are finite and positive; specific heat ratio is "
            "checked against constant-pressure specific heat divided by "
            "constant-volume specific heat"
        ),
        "property_failure_behavior": (
            "both gas maps are evaluated successfully before either HDF5 group "
            "is replaced"
        ),
        "private_thermoprop_api_used": False,
    }


def write_map_group(file: h5py.File, generated_map: GeneratedGasMap) -> None:
    """Replace one generated map group in an already-open HDF5 file."""

    group_name = generated_map.specification.group_name

    if group_name in file:
        del file[group_name]

    map_group = file.create_group(group_name)
    map_group.attrs["kind"] = "map"
    map_group.attrs["name"] = group_name
    map_group.attrs["map_format"] = "rectangular-grid-map-v1"
    map_group.attrs["axis_order"] = json.dumps(["pressure", "enthalpy"])
    map_group.attrs["output_names"] = json.dumps(output_names)
    map_group.attrs["constants"] = json.dumps({})
    map_group.attrs["metadata"] = json.dumps(map_metadata(generated_map))
    map_group.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

    axes_group = map_group.create_group("axes")

    pressure_dataset = axes_group.create_dataset(
        "pressure",
        data=np.asarray(generated_map.pressure_axis, dtype=float),
    )
    pressure_dataset.attrs["name"] = "pressure"
    pressure_dataset.attrs["spacing"] = "linear"
    pressure_dataset.attrs["units"] = "Pa"

    enthalpy_dataset = axes_group.create_dataset(
        "enthalpy",
        data=np.asarray(generated_map.enthalpy_axis, dtype=float),
    )
    enthalpy_dataset.attrs["name"] = "enthalpy"
    enthalpy_dataset.attrs["spacing"] = "linear"
    enthalpy_dataset.attrs["units"] = "J/kg"

    outputs_group = map_group.create_group("outputs")

    for name in output_names:
        dataset = outputs_group.create_dataset(
            name,
            data=np.asarray(generated_map.values[name], dtype=float),
        )
        dataset.attrs["units"] = output_units[name]

    status_group = map_group.create_group("status")
    shape = (
        generated_map.pressure_axis.size,
        generated_map.enthalpy_axis.size,
    )
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


def write_maps(generated_maps: list[GeneratedGasMap]) -> str:
    """Replace both gas maps while preserving every unrelated HDF5 group."""

    with h5py.File(filename, "a") as file:
        for generated_map in generated_maps:
            write_map_group(file, generated_map)

        file.attrs["updated_utc"] = datetime.now(timezone.utc).isoformat()

    return filename


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

    total_points_per_fluid = pressure_count * enthalpy_count
    raw_output_size_mib_per_fluid = (
        total_points_per_fluid * len(output_names) * 8.0 / (1024.0**2)
    )

    print()
    print(f"Target file: {filename}")
    print("Target groups: /nitrogen_hp and /helium_hp")
    print(
        f"Pressure range: {minimum_pressure:.6g} to {maximum_pressure:.6g} Pa "
        f"({minimum_pressure / psia_to_pa:.3f} to "
        f"{maximum_pressure / psia_to_pa:,.1f} psia)"
    )
    print(
        f"Nominal temperature range: {minimum_temperature:.1f} to "
        f"{maximum_temperature:.1f} K"
    )
    print(f"Map shape per fluid: ({pressure_count}, {enthalpy_count})")
    print(f"HP states per fluid: {total_points_per_fluid:,}")
    print(
        f"Raw seven-output payload per fluid: approximately "
        f"{raw_output_size_mib_per_fluid:.2f} MiB"
    )

    # Complete and validate both fluids before mutating the shared HDF5 file.
    generated_maps: list[GeneratedGasMap] = []

    for specification in gas_specifications:
        minimum_enthalpy, maximum_enthalpy = enthalpy_axis_limits(
            specification,
            pressure_axis,
        )
        enthalpy_axis = np.linspace(
            minimum_enthalpy,
            maximum_enthalpy,
            enthalpy_count,
            dtype=float,
        )

        print()
        print(
            f"{specification.group_name}: enthalpy range = "
            f"{minimum_enthalpy:.6g} to {maximum_enthalpy:.6g} J/kg"
        )

        values, phase_counts = precompute_map(
            specification=specification,
            pressure_axis=pressure_axis,
            enthalpy_axis=enthalpy_axis,
        )
        generated_maps.append(
            GeneratedGasMap(
                specification=specification,
                pressure_axis=pressure_axis,
                enthalpy_axis=enthalpy_axis,
                values=values,
                phase_counts=phase_counts,
            )
        )

    print()
    print("Both gas maps validated; replacing their HDF5 groups.")
    output_path = write_maps(generated_maps)

    print()
    print(f"Completed: {output_path}")
    for generated_map in generated_maps:
        print(
            f"Group: /{generated_map.specification.group_name} | "
            f"fluid={generated_map.specification.fluid_name} | "
            f"outputs={', '.join(output_names)}"
        )
    print("All unrelated HDF5 groups were preserved.")
    print()


if __name__ == "__main__":
    main()
