from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from CoolProp.CoolProp import PropsSI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Flight.PropSystem import PropSystem


@dataclass(frozen=True)
class GasGeometry:
    volume: float
    length: float
    diameter: float
    internal_area: float

    @staticmethod
    def axial_mass(mass):
        return (mass,)


@dataclass(frozen=True)
class PropGeometry:
    volume: float
    inner_diameter: float
    cylinder_length: float
    ellipse_ratio: float
    passthrough_diameter: float

    @property
    def internal_area(self):
        return (
            math.pi * self.inner_diameter * self.cylinder_length
            + math.pi * self.inner_diameter**2
        )

    def fill_state(self, liquid_volume):
        fraction = liquid_volume / self.volume
        return {
            "fill_height": fraction
            * (self.cylinder_length + self.inner_diameter / self.ellipse_ratio),
            "liquid_contact_area": self.internal_area * fraction,
            "ullage_contact_area": self.internal_area * (1.0 - fraction),
        }

    @staticmethod
    def axial_mass(liquid_volume, liquid_mass, ullage_mass):
        return (liquid_mass + ullage_mass,)


class GeometrySource:
    def __init__(self, geometry):
        self.geometry = geometry

    def get_fluid_geometry(self):
        return self.geometry


def stored_state(fluid, pressure, temperature, volume):
    density = PropsSI("Dmass", "P", pressure, "T", temperature, fluid)
    energy = PropsSI("Umass", "P", pressure, "T", temperature, fluid)
    mass = density * volume
    return mass, mass * energy


def initialize_tanks(cfg):
    state0 = cfg["prop_system"]["state0"]
    geometry_cfg = cfg["tank_geometry"]
    tanks = {}

    press = state0["press_tank"]
    press_geometry = GasGeometry(**geometry_cfg["press_tank"])
    press["m"], press["U"] = stored_state(
        press["fluid"], press["P"], press["T"], press_geometry.volume
    )
    tanks["press_tank"] = GeometrySource(press_geometry)

    for tank_id in ("ox_tank", "fuel_tank"):
        state = state0[tank_id]
        geometry = PropGeometry(**geometry_cfg[tank_id])
        liquid_volume = geometry.volume * float(state["initial_fill_fraction"])
        ullage_volume = geometry.volume - liquid_volume
        state["m_liq"], state["U_liq"] = stored_state(
            state["fluid"], state["P"], state["T"], liquid_volume
        )
        state["m_ull"], state["U_ull"] = stored_state(
            state["gas_fluid"], state["P"], state["gas_T"], ullage_volume
        )
        tanks[tank_id] = GeometrySource(geometry)
    return tanks


def stored_mass(result):
    return sum(
        float(result.node[node_id]["mass"])
        for node_id in result.td_state
    )


def validate(result, previous_mass, dt, mass_balance_rtol):
    for node_id, state in result.node.items():
        pressure = float(state["P"])
        if not np.isfinite(pressure) or pressure <= 0.0:
            raise RuntimeError(f"Node '{node_id}' has invalid pressure {pressure}")
        if "mass" in state and (
            not np.isfinite(state["mass"]) or state["mass"] < 0.0
        ):
            raise RuntimeError(f"Node '{node_id}' has invalid mass {state['mass']}")

    for branch_id, mdot in result.mdot.items():
        if not np.isfinite(mdot):
            raise RuntimeError(f"Branch '{branch_id}' has invalid mass flow {mdot}")

    propulsion = result.propulsion
    for name in ("thrust", "Pc", "MR", "mdot_ox", "mdot_fuel", "mdot_nozzle"):
        if not np.isfinite(getattr(propulsion, name)):
            raise RuntimeError(f"Propulsion output '{name}' is not finite")

    current_mass = stored_mass(result)
    expected_change = -dt * float(propulsion.mdot_nozzle)
    balance_error = current_mass - previous_mass - expected_change
    tolerance = mass_balance_rtol * max(abs(expected_change), 1.0)
    if abs(balance_error) > tolerance:
        raise RuntimeError(
            f"Fluid mass balance error {balance_error:.6e} kg exceeds {tolerance:.6e} kg"
        )
    return current_mass, balance_error


def trace_row(time_s, result, balance_error):
    propulsion = result.propulsion
    row = {
        "time_s": time_s,
        "thrust_N": propulsion.thrust,
        "Pc_Pa": propulsion.Pc,
        "MR": propulsion.MR,
        "mdot_ox_kg_s": propulsion.mdot_ox,
        "mdot_fuel_kg_s": propulsion.mdot_fuel,
        "mdot_nozzle_kg_s": propulsion.mdot_nozzle,
        "mass_balance_error_kg": balance_error,
    }
    for node_id, state in result.node.items():
        row[f"{node_id}_P_Pa"] = state["P"]
        if "mass" in state:
            row[f"{node_id}_mass_kg"] = state["mass"]
    return row


def run(config_path, duration=None, dt=None, output=None):
    with config_path.open("rb") as stream:
        cfg = tomllib.load(stream)
    simulation = cfg["simulation"]
    duration = float(simulation["duration"] if duration is None else duration)
    dt = float(simulation["dt"] if dt is None else dt)
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("Harness duration and timestep must be positive")
    steps_float = duration / dt
    steps = round(steps_float)
    if not math.isclose(steps_float, steps, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Harness duration must be an integer multiple of dt")

    system = PropSystem(cfg, initialize_tanks(cfg))
    atmosphere = SimpleNamespace(p=float(simulation["ambient_pressure"]))
    initial = system.update(dt=None, atm=atmosphere, heat_flux={}, commit=False)
    previous_mass = stored_mass(initial)
    mass_balance_rtol = float(simulation["mass_balance_rtol"])
    progress_every = int(simulation["progress_every"])
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    rows = []
    started = time.perf_counter()

    for step in range(1, steps + 1):
        time_s = step * dt
        try:
            result = system.update(dt=dt, atm=atmosphere, heat_flux={})
            previous_mass, error = validate(
                result, previous_mass, dt, mass_balance_rtol
            )
        except Exception as failure:
            raise RuntimeError(
                f"Propulsion harness failed at step {step}/{steps}, t={time_s:.1f} s"
            ) from failure
        rows.append(trace_row(time_s, result, error))
        if step % progress_every == 0 or step == steps:
            print(
                f"step {step:>4}/{steps}, t={time_s:>6.1f} s, "
                f"thrust={result.propulsion.thrust:>9.1f} N",
                flush=True,
            )

    output_path = Path(simulation["output"] if output is None else output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - started
    thrust = np.array([row["thrust_N"] for row in rows])
    print(f"PASS: {steps} steps over {duration:.1f} simulated seconds")
    print(f"Wall time: {elapsed:.2f} s")
    print(f"Thrust range: {thrust.min():.1f} to {thrust.max():.1f} N")
    print(f"Final stored fluid mass: {previous_mass:.3f} kg")
    print(f"Trace: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=ROOT / "Configs/propulsion_3min_harness.toml",
    )
    parser.add_argument("--duration", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    run(arguments.config, arguments.duration, arguments.dt, arguments.output)
