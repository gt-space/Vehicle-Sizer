"""Generate rectangular pressure-temperature property tables with CoolProp."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np
from CoolProp import AbstractState
from CoolProp.CoolProp import PQ_INPUTS, PT_INPUTS, PropsSI, iphase_gas, iphase_liquid


LOOKUP_FILE = Path(__file__).with_name("sizer_lookups.h5")


@dataclass(frozen=True)
class PTSpecification:
    fluid: str
    group: str
    pressure_bounds: tuple[float, float]
    temperature_bounds: tuple[float, float]
    pressure_count: int
    temperature_count: int
    phase: str


OUTPUT_UNITS = {
    "density": "kg/m^3",
    "enthalpy": "J/kg",
    "internal_energy": "J/kg",
    "dynamic_viscosity": "Pa*s",
    "conductivity": "W/(m*K)",
    "specific_heat_at_constant_pressure": "J/(kg*K)",
    "specific_heat_at_constant_volume": "J/(kg*K)",
    "specific_heat_ratio": "dimensionless",
}


def _state_outputs(state: AbstractState) -> tuple[float, ...]:
    cp = state.cpmass()
    cv = state.cvmass()
    return (
        state.rhomass(),
        state.hmass(),
        state.umass(),
        state.viscosity(),
        state.conductivity(),
        cp,
        cv,
        cp / cv,
    )


def generate_pt_table(spec: PTSpecification, path: Path = LOOKUP_FILE) -> None:
    """Generate one complete PT map, then replace only its HDF5 group."""

    pressure = np.geomspace(*spec.pressure_bounds, spec.pressure_count)
    temperature = np.linspace(*spec.temperature_bounds, spec.temperature_count)
    names = tuple(OUTPUT_UNITS)
    values = {name: np.empty((pressure.size, temperature.size)) for name in names}
    success = np.ones((pressure.size, temperature.size), dtype=np.bool_)
    state = AbstractState("HEOS", spec.fluid)
    phases = {"gas": iphase_gas, "liquid": iphase_liquid}
    if spec.phase != "auto":
        try:
            state.specify_phase(phases[spec.phase])
        except KeyError as error:
            raise ValueError(f"Unsupported imposed phase {spec.phase!r}") from error

    for i, p_value in enumerate(pressure):
        for j, t_value in enumerate(temperature):
            try:
                state.update(PT_INPUTS, float(p_value), float(t_value))
                outputs = _state_outputs(state)
            except Exception:
                success[i, j] = False
                outputs = (np.nan,) * len(names)
            if (
                not np.all(np.isfinite(outputs))
                or outputs[0] <= 0.0
                or outputs[-1] <= 0.0
            ):
                success[i, j] = False
                outputs = (np.nan,) * len(names)
            for name, value in zip(names, outputs):
                values[name][i, j] = value

    gas_constant = state.gas_constant() / state.molar_mass()
    metadata = {
        "description": f"{spec.fluid} {spec.phase} properties on a PT grid",
        "fluid": spec.fluid,
        "phase": spec.phase,
        "state_inputs": ["pressure", "temperature"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "CoolProp HEOS",
        "extrapolation": "forbidden",
    }
    temporary = f"__new_{spec.group}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as file:
        if temporary in file:
            del file[temporary]
        group = file.create_group(temporary)
        group.attrs["axis_order"] = json.dumps(["pressure", "temperature"])
        group.attrs["output_names"] = json.dumps(list(names))
        group.attrs["metadata"] = json.dumps(metadata)
        group.attrs["constants"] = json.dumps(
            {"specific_gas_constant": float(gas_constant)}
        )
        axes = group.create_group("axes")
        axes.create_dataset("pressure", data=pressure).attrs["units"] = "Pa"
        axes.create_dataset("temperature", data=temperature).attrs["units"] = "K"
        outputs = group.create_group("outputs")
        for name, data in values.items():
            dataset = outputs.create_dataset(
                name, data=data, compression="gzip", shuffle=True
            )
            dataset.attrs["units"] = OUTPUT_UNITS[name]
        status = group.create_group("status")
        status.create_dataset(
            "success",
            data=success,
            compression="gzip",
        )
        if spec.group in file:
            del file[spec.group]
        file.move(temporary, spec.group)
        file.attrs["updated_utc"] = datetime.now(timezone.utc).isoformat()


def generate_saturation_table(
    fluid: str,
    group_name: str,
    pressure_count: int = 600,
    path: Path = LOOKUP_FILE,
) -> None:
    """Generate saturated liquid and vapor properties indexed by pressure."""

    bounds = (
        float(PropsSI("PTRIPLE", fluid)) * 1.001,
        float(PropsSI("PCRIT", fluid)) * 0.999,
    )
    pressure = np.geomspace(*bounds, pressure_count)
    prefixes = ("liquid", "vapor")
    properties = (
        "density",
        "enthalpy",
        "internal_energy",
        "specific_heat_ratio",
        "dynamic_viscosity",
        "conductivity",
        "specific_heat_at_constant_pressure",
        "isobaric_expansion_coefficient",
    )
    values = {"temperature": np.empty(pressure.size)}
    values.update(
        {
            f"{prefix}_{name}": np.empty(pressure.size)
            for prefix in prefixes
            for name in properties
        }
    )
    state = AbstractState("HEOS", fluid)
    for index, p_value in enumerate(pressure):
        for prefix, quality in zip(prefixes, (0.0, 1.0)):
            state.update(PQ_INPUTS, float(p_value), quality)
            density, enthalpy, energy, _, _, cp, cv, gamma = _state_outputs(state)
            values["temperature"][index] = state.T()
            values[f"{prefix}_density"][index] = density
            values[f"{prefix}_enthalpy"][index] = enthalpy
            values[f"{prefix}_internal_energy"][index] = energy
            values[f"{prefix}_specific_heat_ratio"][index] = gamma
            values[f"{prefix}_dynamic_viscosity"][index] = state.viscosity()
            values[f"{prefix}_conductivity"][index] = state.conductivity()
            values[f"{prefix}_specific_heat_at_constant_pressure"][index] = cp
            values[f"{prefix}_isobaric_expansion_coefficient"][index] = (
                state.isobaric_expansion_coefficient()
            )

    temporary = f"__new_{group_name}"
    with h5py.File(path, "a") as file:
        if temporary in file:
            del file[temporary]
        group = file.create_group(temporary)
        group.attrs["axis_order"] = json.dumps(["pressure"])
        group.attrs["output_names"] = json.dumps(list(values))
        group.attrs["metadata"] = json.dumps(
            {
                "description": f"{fluid} liquid-vapor saturation properties",
                "fluid": fluid,
                "state_inputs": ["pressure"],
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "backend": "CoolProp HEOS",
                "extrapolation": "forbidden",
            }
        )
        group.attrs["constants"] = json.dumps(
            {
                "specific_gas_constant": float(
                    state.gas_constant() / state.molar_mass()
                )
            }
        )
        axes = group.create_group("axes")
        axes.create_dataset("pressure", data=pressure).attrs["units"] = "Pa"
        outputs = group.create_group("outputs")
        for name, data in values.items():
            dataset = outputs.create_dataset(name, data=data)
            dataset.attrs["units"] = (
                "K"
                if name == "temperature"
                else "Pa*s"
                if name.endswith("dynamic_viscosity")
                else "W/(m*K)"
                if name.endswith("conductivity")
                else "J/(kg*K)"
                if name.endswith("specific_heat_at_constant_pressure")
                else "dimensionless"
                if name.endswith(("ratio", "coefficient"))
                else "kg/m^3"
                if name.endswith("density")
                else "J/kg"
            )
        status = group.create_group("status")
        status.create_dataset(
            "success", data=np.ones(pressure.size, dtype=np.bool_)
        )
        if group_name in file:
            del file[group_name]
        file.move(temporary, group_name)
        file.attrs["updated_utc"] = datetime.now(timezone.utc).isoformat()
