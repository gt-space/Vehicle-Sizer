from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

import h5py
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from fullplot import Axis, generate_map
from thermoprop import Propellant


# ---------------------------------------------------------------------------
# File and HDF5 group
# ---------------------------------------------------------------------------

# Only /rp1_hp is replaced. Existing groups in sizer_lookups.h5 are preserved.
filename = "sizer_lookups"
group = "rp1_lookup"

psia_to_pa = 6894.757293168361


# ---------------------------------------------------------------------------
# Map range and resolution
# ---------------------------------------------------------------------------

# Both inputs are stored in SI units and use linear axes.
minimum_pressure = 101325.0
maximum_pressure = 650.0 * psia_to_pa

# These temperatures define the RP-1 property region used to construct the map.
minimum_temperature = 250.0
maximum_temperature = 650.0

pressure_count = 100
enthalpy_count = 200

# Each pressure row is evaluated first on a smaller temperature grid. The
# resulting public ThermoProp property curves are then interpolated onto the
# final pressure-enthalpy grid.
temperature_curve_count = 101

# Propellant.enthalpy is comparatively expensive because it performs numerical
# integration. Only these sparse temperature locations call it directly. The
# values between them are reconstructed from public Cp and corrected to the
# public enthalpy values at the anchors.
enthalpy_anchor_count = 11

# Keep sampled states slightly inside the liquid side of saturation.
saturation_temperature_margin = 0.5


# ---------------------------------------------------------------------------
# Multiprocessing
# ---------------------------------------------------------------------------

# Two workers provide multiprocessing without driving every CPU core. Increase
# this to 4 if maximum speed matters more than fan noise.
workers = min(2, os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

# These are the only datasets written under /rp1_hp/outputs.
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
    "description": "RP-1 properties tabulated by pressure and enthalpy",
    "propellant": "RP-1",
    "state_inputs": ["pressure", "enthalpy"],
    "outputs": output_names,
    "pressure_units": "Pa",
    "enthalpy_units": "J/kg",
    "enthalpy_basis": "ThermoProp Propellant public enthalpy property",
    "temperature_range_K": [minimum_temperature, maximum_temperature],
    "invalid_state_behavior": "NaN outside the valid liquid RP-1 region",
    "private_thermoprop_api_used": False,
}


# ---------------------------------------------------------------------------
# Worker-local state
# ---------------------------------------------------------------------------

_worker_rp1: Propellant | None = None
_worker_enthalpy_axis: np.ndarray | None = None


def finite_float(value, name: str) -> float:
    """Return a finite float or raise a useful error."""

    if value is None:
        raise ValueError(f"{name} is unavailable.")

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"{name} is not finite.")

    return value


def initialize_worker(enthalpy_axis: np.ndarray) -> None:
    """Create one reusable public Propellant object in each worker."""

    global _worker_rp1
    global _worker_enthalpy_axis

    _worker_enthalpy_axis = np.asarray(enthalpy_axis, dtype=float)
    _worker_rp1 = Propellant(
        "RP-1",
        temperature=298.15,
        pressure=maximum_pressure,
    )


def set_state(pressure: float, temperature: float) -> None:
    """Update RP-1 through the public pressure-temperature setter."""

    if _worker_rp1 is None:
        raise RuntimeError("The worker RP-1 object has not been initialized.")

    _worker_rp1.pressure_temperature = (
        float(pressure),
        float(temperature),
    )


def maximum_valid_temperature(pressure: float) -> float:
    """Return the highest liquid temperature sampled at one pressure."""

    if _worker_rp1 is None:
        raise RuntimeError("The worker RP-1 object has not been initialized.")

    set_state(pressure, minimum_temperature)

    upper_temperature = maximum_temperature

    backend_maximum = _worker_rp1.maximum_temperature
    if backend_maximum is not None and np.isfinite(backend_maximum):
        upper_temperature = min(
            upper_temperature,
            float(backend_maximum),
        )

    saturation_temperature = _worker_rp1.saturation_temperature
    if saturation_temperature is not None and np.isfinite(saturation_temperature):
        upper_temperature = min(
            upper_temperature,
            float(saturation_temperature) - saturation_temperature_margin,
        )

    return float(upper_temperature)


def empty_row() -> dict[str, np.ndarray]:
    """Return one NaN-filled row for all requested outputs."""

    if _worker_enthalpy_axis is None:
        raise RuntimeError("The worker enthalpy axis has not been initialized.")

    return {
        name: np.full(_worker_enthalpy_axis.size, np.nan, dtype=float)
        for name in output_names
    }


def public_property_curves(
    pressure: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Evaluate the requested public properties versus temperature."""

    if _worker_rp1 is None:
        raise RuntimeError("The worker RP-1 object has not been initialized.")

    upper_temperature = maximum_valid_temperature(pressure)

    if upper_temperature <= minimum_temperature:
        return None

    temperatures = np.linspace(
        minimum_temperature,
        upper_temperature,
        temperature_curve_count,
        dtype=float,
    )

    curves = {
        "density": np.empty(temperatures.size, dtype=float),
        "dynamic_viscosity": np.empty(temperatures.size, dtype=float),
        "conductivity": np.empty(temperatures.size, dtype=float),
        "specific_heat": np.empty(temperatures.size, dtype=float),
    }

    for index, temperature in enumerate(temperatures):
        set_state(pressure, float(temperature))

        curves["density"][index] = finite_float(
            _worker_rp1.density,
            "RP-1 density",
        )
        curves["dynamic_viscosity"][index] = finite_float(
            _worker_rp1.dynamic_viscosity,
            "RP-1 dynamic viscosity",
        )
        curves["conductivity"][index] = finite_float(
            _worker_rp1.conductivity,
            "RP-1 conductivity",
        )
        curves["specific_heat"][index] = finite_float(
            _worker_rp1.specific_heat,
            "RP-1 specific heat",
        )

    return temperatures, curves


def public_enthalpy_curve(
    pressure: float,
    temperatures: np.ndarray,
    specific_heat: np.ndarray,
) -> np.ndarray:
    """Construct h(T,P) using only public Propellant properties.

    At constant pressure, Cp supplies the temperature derivative of enthalpy.
    Sparse calls to the public ``Propellant.enthalpy`` property establish the
    exact ThermoProp reference basis and correct any pressure-dependent drift.
    This avoids calling the expensive enthalpy integration at every map point.
    """

    if _worker_rp1 is None:
        raise RuntimeError("The worker RP-1 object has not been initialized.")

    integrated_cp = np.concatenate(
        (
            np.asarray([0.0], dtype=float),
            cumulative_trapezoid(specific_heat, temperatures),
        )
    )

    anchor_indices = np.unique(
        np.linspace(
            0,
            temperatures.size - 1,
            min(enthalpy_anchor_count, temperatures.size),
            dtype=int,
        )
    )

    anchor_temperatures = temperatures[anchor_indices]
    anchor_corrections = np.empty(anchor_indices.size, dtype=float)

    for anchor_number, curve_index in enumerate(anchor_indices):
        set_state(pressure, float(temperatures[curve_index]))

        public_enthalpy = finite_float(
            _worker_rp1.enthalpy,
            "RP-1 enthalpy",
        )

        anchor_corrections[anchor_number] = (
            public_enthalpy - integrated_cp[curve_index]
        )

    if anchor_indices.size == 1:
        correction = np.full_like(
            temperatures,
            anchor_corrections[0],
        )
    else:
        correction = PchipInterpolator(
            anchor_temperatures,
            anchor_corrections,
            extrapolate=False,
        )(temperatures)

    enthalpy = np.asarray(integrated_cp + correction, dtype=float)

    if not np.all(np.isfinite(enthalpy)):
        raise ValueError("The reconstructed RP-1 enthalpy curve is not finite.")

    return enthalpy


def evaluate_pressure_row(job: tuple[int, float]):
    """Evaluate one complete pressure row in a worker process."""

    row_index, pressure = job
    pressure = float(pressure)
    row = empty_row()

    try:
        curve_data = public_property_curves(pressure)

        if curve_data is None:
            return row_index, row, 0, None

        temperatures, curves = curve_data
        enthalpy_curve = public_enthalpy_curve(
            pressure,
            temperatures,
            curves["specific_heat"],
        )

        # h(T,P) should increase with temperature. Remove any duplicate or
        # nonincreasing values caused only by numerical noise before inversion.
        keep = np.concatenate(
            (
                np.asarray([True]),
                np.diff(enthalpy_curve) > 0.0,
            )
        )

        temperatures = temperatures[keep]
        enthalpy_curve = enthalpy_curve[keep]
        curves = {
            name: values[keep]
            for name, values in curves.items()
        }

        if enthalpy_curve.size < 2:
            raise ValueError("Too few monotonic RP-1 enthalpy points.")

        if _worker_enthalpy_axis is None:
            raise RuntimeError("The worker enthalpy axis has not been initialized.")

        valid = (
            (_worker_enthalpy_axis >= enthalpy_curve[0])
            & (_worker_enthalpy_axis <= enthalpy_curve[-1])
        )

        if not np.any(valid):
            return row_index, row, 0, None

        valid_enthalpy = _worker_enthalpy_axis[valid]

        temperature_from_enthalpy = PchipInterpolator(
            enthalpy_curve,
            temperatures,
            extrapolate=False,
        )

        target_temperature = np.asarray(
            temperature_from_enthalpy(valid_enthalpy),
            dtype=float,
        )

        row["temperature"][valid] = target_temperature

        for name in (
            "density",
            "dynamic_viscosity",
            "conductivity",
            "specific_heat",
        ):
            property_from_temperature = PchipInterpolator(
                temperatures,
                curves[name],
                extrapolate=False,
            )

            row[name][valid] = property_from_temperature(target_temperature)

        return row_index, row, int(np.count_nonzero(valid)), None

    except Exception as exc:
        return row_index, row, 0, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Precompute in parallel, then write through FullPlot
# ---------------------------------------------------------------------------


def precompute_map(
    pressure_axis: np.ndarray,
    enthalpy_axis: np.ndarray,
) -> dict[str, np.ndarray]:
    """Precompute every output array with multiprocessing."""

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
        f"{group}: each row uses {temperature_curve_count} public PT states "
        f"and {enthalpy_anchor_count} public enthalpy calls"
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
            row_index, row, row_valid_points, error = future.result()

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

            if error is not None:
                message += f" | pressure={pressure_axis[row_index]:.6g} Pa | {error}"

            print(message, flush=True)

    return values


def write_map(
    axes: list[Axis],
    values: dict[str, np.ndarray],
) -> str:
    """Write the precomputed arrays using FullPlot's map format."""

    pressure_axis = np.asarray(axes[0].values, dtype=float)
    enthalpy_axis = np.asarray(axes[1].values, dtype=float)

    pressure_index = {
        float(value): index
        for index, value in enumerate(pressure_axis)
    }
    enthalpy_index = {
        float(value): index
        for index, value in enumerate(enthalpy_axis)
    }

    total_points = pressure_axis.size * enthalpy_axis.size
    write_count = 0
    write_start = time.perf_counter()

    def lookup(pressure: float, enthalpy: float):
        nonlocal write_count

        i = pressure_index[float(pressure)]
        j = enthalpy_index[float(enthalpy)]

        write_count += 1

        if write_count % 5000 == 0 or write_count == total_points:
            elapsed = time.perf_counter() - write_start
            rate = write_count / elapsed if elapsed > 0.0 else 0.0
            print(
                f"{group}: writing {write_count:,}/{total_points:,} "
                f"({write_count / total_points:.1%}) at {rate:,.0f} points/s",
                flush=True,
            )

        return {
            name: float(values[name][i, j])
            for name in output_names
        }

    # The raw output arrays are less than 1 MB, so compression is intentionally
    # disabled. This makes the write faster and avoids unnecessary CPU usage.
    return generate_map(
        filename=filename,
        group=group,
        axes=axes,
        outputs=output_names,
        evaluate=lookup,
        metadata=metadata,
        overwrite=True,
        resume=False,
        raise_errors=True,
        compression=None,
        compression_opts=None,
        flush_every=5000,
    )


# ---------------------------------------------------------------------------
# HDF5 metadata
# ---------------------------------------------------------------------------


def apply_output_units(path: str) -> None:
    """Attach units to the five output datasets."""

    with h5py.File(path, "a") as file:
        outputs = file[group]["outputs"]

        for name, units in output_units.items():
            outputs[name].attrs["units"] = units


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def enthalpy_axis_limits() -> tuple[float, float]:
    """Get the common H-axis limits from public RP-1 states at 650 psia."""

    rp1 = Propellant(
        "RP-1",
        temperature=minimum_temperature,
        pressure=maximum_pressure,
    )

    minimum_enthalpy = finite_float(
        rp1.enthalpy,
        "RP-1 minimum map enthalpy",
    )

    rp1.pressure_temperature = (
        maximum_pressure,
        maximum_temperature,
    )

    maximum_enthalpy = finite_float(
        rp1.enthalpy,
        "RP-1 maximum map enthalpy",
    )

    if maximum_enthalpy <= minimum_enthalpy:
        raise ValueError("The RP-1 enthalpy axis is not increasing.")

    return minimum_enthalpy, maximum_enthalpy


def main() -> None:
    minimum_enthalpy, maximum_enthalpy = enthalpy_axis_limits()

    axes = [
        Axis.linear(
            "pressure",
            start=minimum_pressure,
            stop=maximum_pressure,
            count=pressure_count,
            units="Pa",
        ),
        Axis.linear(
            "enthalpy",
            start=minimum_enthalpy,
            stop=maximum_enthalpy,
            count=enthalpy_count,
            units="J/kg",
        ),
    ]

    pressure_axis = np.asarray(axes[0].values, dtype=float)
    enthalpy_axis = np.asarray(axes[1].values, dtype=float)

    total_points = pressure_count * enthalpy_count
    raw_output_mb = total_points * len(output_names) * 8.0 / 1024.0**2

    print()
    print(f"Target file: {filename}.h5")
    print(f"Target group: /{group}")
    print(
        f"Pressure: {minimum_pressure:.6g} to {maximum_pressure:.6g} Pa "
        f"({maximum_pressure / psia_to_pa:.1f} psia)"
    )
    print(
        f"Enthalpy: {minimum_enthalpy:.6g} to "
        f"{maximum_enthalpy:.6g} J/kg"
    )
    print(
        f"H-axis temperature basis: {minimum_temperature:.1f} to "
        f"{maximum_temperature:.1f} K at 650 psia"
    )
    print(f"Map shape: ({pressure_count}, {enthalpy_count})")
    print(f"Raw five-output payload: approximately {raw_output_mb:.2f} MB")

    values = precompute_map(pressure_axis, enthalpy_axis)

    print()
    print(f"{group}: thermophysical evaluation complete; writing map")
    print()

    generated_path = write_map(axes, values)
    apply_output_units(generated_path)

    print()
    print(f"Completed: {generated_path}")
    print(f"Group: /{group}")
    print(f"Outputs: {', '.join(output_names)}")


if __name__ == "__main__":
    main()
