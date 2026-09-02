from __future__ import annotations

import argparse
import os
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="vehicle-sizer-mpl-"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from Flight.PropSystem import PropSystem


@dataclass(frozen=True)
class PressTankGeometry:
    volume: float
    length: float
    diameter: float
    internal_area: float

    @staticmethod
    def axial_mass(mass):
        return [mass]


@dataclass(frozen=True)
class PropTankGeometry:
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
        area = self.internal_area
        return {
            "fill_height": fraction
            * (self.cylinder_length + self.inner_diameter / self.ellipse_ratio),
            "liquid_contact_area": area * fraction,
            "ullage_contact_area": area * (1.0 - fraction),
        }

    @staticmethod
    def axial_mass(liquid_volume, liquid_mass, ullage_mass):
        return [liquid_mass + ullage_mass]


class TankGeometry:
    """Expose an immutable geometry through the interface PropSystem expects."""

    def __init__(self, geometry):
        self.geometry = geometry

    def get_fluid_geometry(self):
        return self.geometry


def stored_state(fluid: str, pressure: float, temperature: float, volume: float):
    rho = PropsSI("Dmass", "P", pressure, "T", temperature, fluid)
    u = PropsSI("Umass", "P", pressure, "T", temperature, fluid)
    mass = rho * volume
    return mass, mass * u


def initialize_tanks(cfg):
    state0 = cfg["prop_system"]["state0"]
    geometry_cfg = cfg["tank_geometry"]
    tanks = {}

    press = state0["press_tank"]
    press_geometry = PressTankGeometry(**geometry_cfg["press_tank"])
    press["m"], press["U"] = stored_state(
        press["fluid"], press["P"], press["T"], press_geometry.volume
    )
    tanks["press_tank"] = TankGeometry(press_geometry)

    for tank_id in ("ox_tank", "fuel_tank"):
        state = state0[tank_id]
        tank_volume = float(geometry_cfg[tank_id]["volume"])
        liquid_volume = tank_volume * float(state["initial_fill_fraction"])
        ullage_volume = tank_volume - liquid_volume

        state["m_liq"], state["U_liq"] = stored_state(
            state["fluid"], state["P"], state["T"], liquid_volume
        )
        state["m_ull"], state["U_ull"] = stored_state(
            state["gas_fluid"], state["P"], state["gas_T"], ullage_volume
        )
        tanks[tank_id] = TankGeometry(
            PropTankGeometry(**geometry_cfg[tank_id])
        )

    return tanks


def run(config_path: Path):
    with config_path.open("rb") as stream:
        cfg = tomllib.load(stream)

    system = PropSystem(cfg, initialize_tanks(cfg))
    sim = cfg["simulation"]
    dt = float(sim["dt"])
    steps = round(float(sim["duration"]) / dt)
    atmosphere = SimpleNamespace(p=float(sim["ambient_pressure"]))

    time = []
    thrust = []
    pressure = {node_id: [] for node_id in system.network.nodes}

    for step in range(steps):
        result = system.update(dt=dt, atm=atmosphere, heat_flux={})
        time.append((step + 1) * dt)
        thrust.append(result["propulsion"]["thrust"])
        for node_id in pressure:
            pressure[node_id].append(result["node"][node_id]["P"])

    output = Path(sim["output"])
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, (pressure_axis, copv_axis, thrust_axis) = plt.subplots(
        3, 1, figsize=(11, 10), sharex=True, constrained_layout=True
    )

    for node_id, values in pressure.items():
        axis = copv_axis if node_id == "press_tank" else pressure_axis
        axis.plot(time, [value / 1.0e6 for value in values], label=node_id)
    pressure_axis.set_ylabel("System pressure [MPa]")
    pressure_axis.grid(alpha=0.3)
    pressure_axis.legend(ncol=3, fontsize=8)
    copv_axis.set_ylabel("COPV pressure [MPa]")
    copv_axis.grid(alpha=0.3)
    copv_axis.legend(fontsize=8)

    thrust_lbf = [value / 4.4482216152605 for value in thrust]
    thrust_axis.plot(time, thrust_lbf, label="computed thrust")
    thrust_axis.axhline(2500.0, color="black", linestyle="--", label="target")
    thrust_axis.set_xlabel("Time [s]")
    thrust_axis.set_ylabel("Thrust [lbf]")
    thrust_axis.grid(alpha=0.3)
    thrust_axis.legend()

    figure.suptitle("2,500 lbf Pressure-Fed Propulsion Simulation, MR = 2.0")
    figure.savefig(output, dpi=180)
    print(f"Saved {output}")
    print(f"Final thrust: {thrust_lbf[-1]:.1f} lbf")
    print(f"Final MR: {result['propulsion']['MR']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=ROOT / "Configs/propulsion_2500lbf_mr2.toml",
    )
    run(parser.parse_args().config)
