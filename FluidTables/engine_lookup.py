from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import contextlib
import io
import json
import os
import time
import warnings

import h5py
import numpy as np

from fullplot import Axis, generate_map


# ---------------------------------------------------------------------------
# User options
# ---------------------------------------------------------------------------

filename = "sizer_lookups"
group = "engine_lookup"

psia_to_pa = 6894.75728

use_multiprocessing = True
map_workers = min(4, os.cpu_count() or 1)
map_chunksize = 250
map_progress_every = 5000

metadata = {
    "description": "RocketCEA lookup table for chamber and nozzle performance parameters",
}


# ---------------------------------------------------------------------------
# Worker-local RocketCEA object
# ---------------------------------------------------------------------------

cea_obj = None


def _worker_init():
    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in scalar divide",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered",
        category=RuntimeWarning,
    )


def get_cea_obj():
    global cea_obj

    if cea_obj is None:
        with contextlib.redirect_stdout(io.StringIO()):
            from rocketcea.cea_obj_w_units import CEA_Obj

            cea_obj = CEA_Obj(
                oxName="LOX",
                fuelName="RP-1",
                temperature_units="degK",
                cstar_units="m/sec",
                specific_heat_units="kJ/kg degK",
                sonic_velocity_units="m/s",
                enthalpy_units="J/kg",
                density_units="kg/m^3",
                pressure_units="Pa",
            )

    return cea_obj


# ---------------------------------------------------------------------------
# RocketCEA helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def get_chamber_properties(Pc, mr):
    cea = get_cea_obj()

    cstar = float(cea.get_Cstar(Pc, mr))
    chamber_temperature = float(cea.get_Tcomb(Pc, mr))
    chamber_density = float(cea.get_Chamber_Density(Pc, mr))

    chamber_molecular_weight, chamber_gamma = cea.get_Chamber_MolWt_gamma(
        Pc,
        mr,
    )

    chamber_molecular_weight = float(chamber_molecular_weight)
    chamber_gamma = float(chamber_gamma)

    return (
        chamber_temperature,
        chamber_density,
        chamber_gamma,
        chamber_molecular_weight,
        cstar,
    )


@lru_cache(maxsize=None)
def get_nozzle_internal(Pc, mr, eps, nfz):
    cea = get_cea_obj()
    nfz = int(nfz)

    if nfz == 0:
        exit_pressure = Pc / cea.get_PcOvPe(Pc, mr, eps)
        exit_mach_number = cea.get_MachNumber(Pc, mr, eps)
        _, _, exit_sonic_velocity = cea.get_SonicVelocities(Pc, mr, eps)

    elif nfz == 1:
        exit_pressure = Pc / cea.get_PcOvPe(Pc, mr, eps, 1)
        exit_mach_number = cea.get_MachNumber(Pc, mr, eps, 1)
        _, _, exit_sonic_velocity = cea.get_SonicVelocities(Pc, mr, eps, 1)

    elif nfz == 2:
        exit_pressure = Pc / cea.get_PcOvPe(Pc, mr, eps, 1, 1)
        exit_mach_number = cea.get_MachNumber(Pc, mr, eps, 1, 1)
        _, _, exit_sonic_velocity = cea.get_SonicVelocities(Pc, mr, eps, 1, 1)

    else:
        raise ValueError(f"Unsupported nfz value: {nfz}")

    exit_velocity = exit_mach_number * exit_sonic_velocity

    return (
        float(exit_pressure),
        float(exit_mach_number),
        float(exit_velocity),
    )


def get_thrust_coefficient(Pc, Pamb, eps, pe, ue, cstar):
    thrust_coefficient = ue / cstar + eps * (pe - Pamb) / Pc
    return float(thrust_coefficient)


def finite_or_nan(value):
    value = float(value)

    if np.isfinite(value):
        return value

    return np.nan


# ---------------------------------------------------------------------------
# Map function
# ---------------------------------------------------------------------------

def engine_map(
    chamber_pressure,
    mixture_ratio,
    expansion_ratio,
    ambient_pressure,
    nfz,
):
    chamber_pressure = float(chamber_pressure)
    mixture_ratio = float(mixture_ratio)
    expansion_ratio = float(expansion_ratio)
    ambient_pressure = float(ambient_pressure)
    nfz = int(nfz)

    try:
        (
            chamber_temperature,
            chamber_density,
            chamber_gamma,
            chamber_molecular_weight,
            characteristic_velocity,
        ) = get_chamber_properties(
            chamber_pressure,
            mixture_ratio,
        )
    except Exception:
        chamber_temperature = np.nan
        chamber_density = np.nan
        chamber_gamma = np.nan
        chamber_molecular_weight = np.nan
        characteristic_velocity = np.nan

    try:
        (
            exit_pressure,
            exit_mach_number,
            exit_velocity,
        ) = get_nozzle_internal(
            chamber_pressure,
            mixture_ratio,
            expansion_ratio,
            nfz,
        )
    except Exception:
        exit_pressure = np.nan
        exit_mach_number = np.nan
        exit_velocity = np.nan

    try:
        thrust_coefficient = get_thrust_coefficient(
            Pc=chamber_pressure,
            Pamb=ambient_pressure,
            eps=expansion_ratio,
            pe=exit_pressure,
            ue=exit_velocity,
            cstar=characteristic_velocity,
        )
    except Exception:
        thrust_coefficient = np.nan

    return {
        "chamber_temperature": finite_or_nan(chamber_temperature),
        "chamber_density": finite_or_nan(chamber_density),
        "chamber_gamma": finite_or_nan(chamber_gamma),
        "chamber_molecular_weight": finite_or_nan(chamber_molecular_weight),
        "exit_pressure": finite_or_nan(exit_pressure),
        "exit_mach_number": finite_or_nan(exit_mach_number),
        "exit_velocity": finite_or_nan(exit_velocity),
        "characteristic_velocity": finite_or_nan(characteristic_velocity),
        "thrust_coefficient": finite_or_nan(thrust_coefficient),
    }


# ---------------------------------------------------------------------------
# Multiprocessing map helper
# ---------------------------------------------------------------------------

def _evaluate_map_job(job):
    evaluate, axis_names, axis_values, constants, index = job

    inputs = dict(constants)

    for name, values, i in zip(axis_names, axis_values, index):
        inputs[name] = float(values[i])

    return index, evaluate(**inputs)


def generate_map_mp(
    filename,
    group,
    axes,
    evaluate,
    constants=None,
    workers=4,
    chunksize=250,
    progress_every=5000,
    **kwargs,
):
    constants = {} if constants is None else dict(constants)

    axis_names = [axis.name for axis in axes]
    axis_values = [axis.values for axis in axes]
    shape = tuple(len(values) for values in axis_values)
    total_points = int(np.prod(shape))
    output_names = tuple(kwargs["outputs"])
    output_data = {
        name: np.empty(shape, dtype=float)
        for name in output_names
    }

    def iter_jobs():
        for index in np.ndindex(shape):
            yield (
                evaluate,
                axis_names,
                axis_values,
                constants,
                index,
            )

    print()
    print(f"{group}: evaluating {total_points:,} map points with {workers} workers...")
    print(f"{group}: shape = {shape}")
    print(f"{group}: chunksize = {chunksize}")
    print()

    start_time = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
    ) as pool:
        for counter, (index, values) in enumerate(
            pool.map(_evaluate_map_job, iter_jobs(), chunksize=chunksize),
            start=1,
        ):
            for name in output_names:
                output_data[name][index] = values[name]

            if counter % progress_every == 0 or counter == total_points:
                elapsed = time.perf_counter() - start_time
                rate = counter / elapsed if elapsed > 0 else 0.0
                remaining = total_points - counter
                eta = remaining / rate if rate > 0 else np.nan

                print(
                    f"{group}: completed "
                    f"{counter:,} / {total_points:,} | "
                    f"{counter / total_points:.2%} | "
                    f"rate = {rate:,.1f} points/s | "
                    f"elapsed = {elapsed / 60:.1f} min | "
                    f"eta = {eta / 60:.1f} min",
                    flush=True,
                )

    print()
    print(f"{group}: writing map in bulk...")
    path = h5_filename(filename)
    with h5py.File(path, "a") as file:
        if group in file:
            del file[group]
        map_group = file.create_group(group)
        map_group.attrs.update({
            "name": group,
            "kind": "map",
            "map_format": "rectangular-grid-map-v1",
            "axis_order": json.dumps(axis_names),
            "output_names": json.dumps(output_names),
            "metadata": json.dumps(kwargs.get("metadata", {})),
            "constants": json.dumps(constants),
        })
        axes_group = map_group.create_group("axes")
        for axis in axes:
            dataset = axes_group.create_dataset(axis.name, data=axis.values)
            dataset.attrs.update({
                "name": axis.name,
                "spacing": axis.spacing,
                "units": axis.units,
            })
        outputs_group = map_group.create_group("outputs")
        for name in output_names:
            outputs_group.create_dataset(name, data=output_data[name])
    return path


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def h5_filename(filename):
    filename = str(filename)

    if not filename.endswith(".h5"):
        filename += ".h5"

    return filename


def file_size_mb(filename):
    filename = h5_filename(filename)

    if not os.path.exists(filename):
        return 0.0

    return os.path.getsize(filename) / (1024.0 * 1024.0)


def shrink_fullplot_map(filename, group):
    filename = h5_filename(filename)

    print()
    print(f"{group}: shrinking final HDF5 map...")
    print(f"{group}: size before shrink = {file_size_mb(filename):.2f} MB")
    print()

    start_time = time.perf_counter()

    with h5py.File(filename, "a") as file:
        map_group = file[group]

        if "status" in map_group:
            print(f"{group}: deleting status group")
            del map_group["status"]

        outputs_group = map_group["outputs"]
        output_names = list(outputs_group.keys())

        for counter, output_name in enumerate(output_names, start=1):
            print(
                f"{group}: compressing output "
                f"{counter} / {len(output_names)}: {output_name}",
                flush=True,
            )

            data = outputs_group[output_name][()].astype(np.float32)

            del outputs_group[output_name]

            outputs_group.create_dataset(
                output_name,
                data=data,
                dtype="f4",
                chunks=True,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fillvalue=np.nan,
            )

            file.flush()

    elapsed = time.perf_counter() - start_time

    print()
    print(f"{group}: shrink complete")
    print(f"{group}: size after shrink = {file_size_mb(filename):.2f} MB")
    print(f"{group}: shrink elapsed = {elapsed / 60:.1f} min")
    print()


# ---------------------------------------------------------------------------
# Map definition
# ---------------------------------------------------------------------------

axes = [
    Axis.linear(
        "chamber_pressure",
        start=50 * psia_to_pa,
        stop=1500 * psia_to_pa,
        count=50,
        units="Pa",
    ),
    Axis.linear(
        "mixture_ratio",
        start=0.1,
        stop=10.0,
        count=61,
    ),
    Axis.linear(
        "expansion_ratio",
        start=1.0,
        stop=10.0,
        count=25,
    ),
    Axis.values(
        "ambient_pressure",
        values=[
            0.01 * psia_to_pa,
            3.0 * psia_to_pa,
            7.5 * psia_to_pa,
            14.7 * psia_to_pa,
            15.0 * psia_to_pa,
        ],
        units="Pa",
    ),
    Axis.values(
        "nfz",
        values=[0, 1, 2],
    ),
]

output_names = [
    "chamber_temperature",
    "chamber_density",
    "chamber_gamma",
    "chamber_molecular_weight",
    "exit_pressure",
    "exit_mach_number",
    "exit_velocity",
    "characteristic_velocity",
    "thrust_coefficient",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    map_generator = generate_map_mp if use_multiprocessing else generate_map

    map_options = {}

    if use_multiprocessing:
        map_options = {
            "workers": map_workers,
            "chunksize": map_chunksize,
            "progress_every": map_progress_every,
        }

    print()
    print(f"{group}: generating {h5_filename(filename)}")
    print(f"{group}: final axes shape = {tuple(len(axis.values) for axis in axes)}")
    print(f"{group}: final points = {int(np.prod([len(axis.values) for axis in axes])):,}")
    print()

    map_generator(
        filename=filename,
        group=group,
        axes=axes,
        outputs=output_names,
        metadata=metadata,
        evaluate=engine_map,
        overwrite=True,
        resume=False,
        raise_errors=True,

        # Important:
        # Write the FullPlot map quickly first. Do not compress here.
        # The shrink step converts to f4 and compresses once at the end.
        compression=None,
        compression_opts=None,
        flush_every=50000,

        **map_options,
    )

    shrink_fullplot_map(filename, group)

    print()
    print(f"{group}: done")
    print(f"{group}: wrote {h5_filename(filename)}")
    print()
