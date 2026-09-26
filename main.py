from __future__ import annotations

import argparse
import time
import csv
from pathlib import Path

import numpy as np

from Configs.loader import load_config
from simulation import ROOT, project_path, property_sources, simulate


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
            "velocity_m_s": kin.vz,  # replaced v with vertical velocity vz
            "acceleration_m_s2": forces["acceleration"],
            "angle_of_attack_deg": np.degrees(kin.alpha),
            "angular_rate_rad_s": kin.q,  # replaced w with pitch rate q
            "mass_kg": mass["total_mass"],
            "cg_m": mass["cg"],
            "Iyy_kg_m2": mass["Iyy"],  # replaced Ixx with pitch inertia, Iyy
            "ambient_pressure_Pa": atmosphere.p,
            "mach": atmosphere.Ma,
            "dynamic_pressure_Pa": atmosphere.q,
            "Cd": aero.Cd,
            "drag_N": forces["drag"],
            "Ca": aero.Ca,
            "axial_aero_force_N": aero.A,
            "Cn": aero.Cn,
            "normal_force_N": aero.N,
            "cp_m": aero.cp,
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
        for branch_id, branch in fluids.branch.items():
            if "opening_fraction" in branch:
                for field in ("opening_fraction", "pressure_error", "max_mdot"):
                    row[f"{branch_id}_{field}"] = branch[field]
                row[f"{branch_id}_mdot"] = fluids.mdot[branch_id]
            if "is_open" in branch:
                row[f"{branch_id}_switch_count"] = fluids.event_counts.get(
                    f"branch:{branch_id}:switch", 0
                )
                row[f"{branch_id}_switches"] = sum(
                    event["kind"] == "branch" and event["component"] == branch_id
                    and "is_open" in event for event in fluids.events
                )
        rows.append(row)
    return rows


def write_history(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def write_events(history: list, path: Path) -> None:
    """Save substep events without downsampling them to the flight output rate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "time_s", "kind", "component", "event", "count", "was_open", "is_open"
        ))
        writer.writeheader()
        writer.writerows(event for state in history for event in state["plant"].fluids.events)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=ROOT / "Configs" / "flight_candidate_350_regulator.yaml",
    )
    parser.add_argument("--dt", type=float)
    parser.add_argument("--t-end", type=float)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.dt is not None:
        cfg["simulation"]["dt"] = args.dt
    if args.t_end is not None:
        cfg["simulation"]["t_end"] = args.t_end

    setup_started = time.perf_counter()
    print("Loading models and running simulation...", flush=True)
    started = time.perf_counter()
    last_report = None

    def report_progress(kin):
        nonlocal last_report
        now = time.perf_counter()
        if last_report is not None and now - last_report < 5.0:
            return
        limit = float(cfg["simulation"]["t_end"])
        percent = min(100.0, 100.0 * kin.t / limit) if limit > 0 else 0.0
        print(f"Simulation: t={kin.t:.3f}/{limit:g} s "
              f"({percent:.1f}% of time limit), altitude={kin.h:.1f} m, "
              f"elapsed={now - started:.1f} s [last completed step]", flush=True)
        last_report = now

    result = simulate(cfg, record_history=True, compute_loads=True, progress=report_progress)
    history = result.history
    if not history:
        if result.termination in {"infeasible_initial_design", "infeasible_operating_state"}:
            failures = [
                f"  {name}: margin={record.margin:.6g} {record.units} (must be >= 0)"
                for name, record in result.constraint_records.items()
                if record.required and record.margin is not None and record.margin < 0
            ]
            raise SystemExit("\n".join([
                f"Simulation rejected: {result.termination}",
                f"Config: {args.config}",
                *failures,
            ]))
        raise RuntimeError(f"Flight simulation returned no time steps ({result.termination})")
    print(f"Propagation finished at t={history[-1]['kinematics'].t:.3f} s "
          f"in {time.perf_counter() - started:.1f} s. Writing history and plots...", flush=True)

    rows = history_rows(history)
    output_path = project_path(cfg["simulation"]["output"])
    plot_path = project_path(cfg["simulation"]["plot"])
    write_history(rows, output_path)
    events_path = output_path.with_name(output_path.stem + "_events.csv")
    write_events(history, events_path)
    from flight_plots import plot_flight
    plots = plot_flight(history, rows, plot_path)
    print(f"Run complete: {time.perf_counter() - setup_started:.1f} s total.", flush=True)

    print(f"Max altitude reached: {result.max_altitude:.1f} m")
    print(f"Apogee: {result.apogee:.1f} m" if result.apogee_reached else "Apogee: not reached")
    print(f"History: {output_path}")
    print(f"Events: {events_path}")
    for name, path in plots.items():
        print(f"{name.replace('_', ' ').title()}: {path}")


if __name__ == "__main__":
    main()
