from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Configs.loader import load_config
from Flight.Flight import FlightSim
from Flight.PropSystem import PropSystem
from Flight.environment import Environment
from Flight.flight_forces import Aero
from FluidProperties.PropertyModels import (
    CEAPropertySource,
    CoolPropPropertySource,
    TableCombustionPropertySource,
    TablePureFluidPropertySource,
)
from Vehicle.Engine import Engine
from Vehicle.Vehicle import Vehicle


ROOT = Path(__file__).resolve().parent


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def property_sources(cfg: dict):
    """Construct the two property interfaces selected by the config."""

    pure_cfg = cfg["property_models"]["pure_fluid"]
    if pure_cfg["source"] == "coolprop":
        pure = CoolPropPropertySource()
    elif pure_cfg["source"] == "table":
        table_cfg = pure_cfg["table"]
        pure = TablePureFluidPropertySource(
            project_path(table_cfg["lookup_file"]),
            table_cfg["fluids"],
        )
    else:
        raise ValueError(f"Unknown pure-fluid source {pure_cfg['source']!r}")

    combustion_cfg = cfg["property_models"]["combustion"]
    if combustion_cfg["source"] == "cea":
        from rocketcea.cea_obj_w_units import CEA_Obj

        engine = cfg["engine"]
        cea = CEA_Obj(
            oxName=engine["oxidizer"],
            fuelName=engine["fuel"],
            pressure_units="Pa",
            cstar_units="m/s",
            temperature_units="K",
            enthalpy_units="J/kg",
            density_units="kg/m^3",
            specific_heat_units="J/kg-K",
        )
        combustion = CEAPropertySource(cea)
    elif combustion_cfg["source"] == "table":
        table_cfg = combustion_cfg["table"]
        combustion = TableCombustionPropertySource(
            project_path(table_cfg["lookup_file"]),
            int(table_cfg["nfz"]),
        )
    else:
        raise ValueError(
            f"Unknown combustion source {combustion_cfg['source']!r}"
        )
    return pure, combustion


def history_rows(history: list) -> list[dict]:
    """Flatten synchronized endpoint states for CSV output."""

    rows = []
    for state in history:
        kin = state["kinematics"]
        atmosphere = state["atmosphere"]
        aero = state["plant"].aero
        propulsion = state["plant"].fluids.propulsion
        fluids = state["plant"].fluids
        forces = state["forces"]
        mass = state["mass_properties"]
        row = {
            "time_s": kin.t,
            "altitude_m": kin.h,
            "velocity_m_s": kin.v,
            "acceleration_m_s2": forces["acceleration"],
            "mass_kg": mass["total_mass"],
            "cg_m": mass["cg"],
            "Ixx_kg_m2": mass["Ixx"],
            "ambient_pressure_Pa": atmosphere.p,
            "mach": atmosphere.Ma,
            "dynamic_pressure_Pa": atmosphere.q,
            "Cd": aero.Cd,
            "drag_N": forces["drag"],
            "thrust_N": propulsion.thrust,
            "chamber_pressure_Pa": propulsion.Pc,
            "mixture_ratio": propulsion.MR,
            "oxidizer_mdot_kg_s": propulsion.mdot_ox,
            "fuel_mdot_kg_s": propulsion.mdot_fuel,
            "nozzle_mdot_kg_s": propulsion.mdot_nozzle,
            "engine_mode": propulsion.mode,
            "engine_on": state["engine_on"],
            "on_rail": state["on_rail"],
        }
        row.update(
            {
                f"{node_id}_pressure_Pa": node["P"]
                for node_id, node in fluids.node.items()
                if "P" in node
            }
        )
        row.update(
            {
                f"{branch_id}_open": branch["is_open"]
                for branch_id, branch in fluids.branch.items()
                if "is_open" in branch
            }
        )
        rows.append(row)
    return rows


def write_history(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def plot_history(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = [row["time_s"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes[0, 0].plot(time, [row["altitude_m"] for row in rows])
    axes[0, 0].set_ylabel("Altitude [m]")
    axes[0, 1].plot(time, [row["velocity_m_s"] for row in rows])
    axes[0, 1].set_ylabel("Velocity [m/s]")
    axes[1, 0].plot(time, [row["thrust_N"] for row in rows])
    axes[1, 0].set_ylabel("Thrust [N]")
    axes[1, 1].plot(time, [row["chamber_pressure_Pa"] for row in rows])
    axes[1, 1].set_ylabel("Chamber pressure [Pa]")
    for axis in axes[-1]:
        axis.set_xlabel("Time [s]")
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=ROOT / "Configs" / "flight_2500lbf_mr2.yaml",
    )
    parser.add_argument("--dt", type=float)
    parser.add_argument("--t-end", type=float)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.dt is not None:
        cfg["simulation"]["dt"] = args.dt
    if args.t_end is not None:
        cfg["simulation"]["t_end"] = args.t_end

    pure_properties, combustion_properties = property_sources(cfg)
    vehicle = Vehicle(cfg, pure_properties)
    propulsion = PropSystem(
        cfg,
        vehicle.tanks,
        fluid_properties=pure_properties,
        combustion_properties=combustion_properties,
    )
    engine = Engine(
        mass=float(cfg["engine"]["mass"]),
        length=float(cfg["engine"]["length"]),
        exit_area=propulsion.exit_area,
    )
    vehicle.build(engine)

    environment_cfg = cfg["environment"]
    environment = Environment(
        h_max=float(environment_cfg["max_altitude"]),
        dh=float(environment_cfg["altitude_step"]),
    )
    aero_cfg = dict(cfg["aero"])
    aero_cfg["cd_table"] = project_path(aero_cfg["cd_table"])
    simulation = FlightSim(
        cfg,
        environment,
        Aero(aero_cfg),
        propulsion,
        vehicle,
    )
    history = simulation.run(
        h0=float(cfg["launch"]["altitude"]),
        v0=float(cfg["launch"]["velocity"]),
    )
    if not history:
        raise RuntimeError("Flight simulation returned no time steps")

    rows = history_rows(history)
    output_path = project_path(cfg["simulation"]["output"])
    plot_path = project_path(cfg["simulation"]["plot"])
    write_history(rows, output_path)
    plot_history(rows, plot_path)

    apogee = max(row["altitude_m"] for row in rows)
    print(f"Apogee: {apogee:.1f} m")
    print(f"History: {output_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
