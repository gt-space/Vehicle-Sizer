from __future__ import annotations

import argparse
import time
import csv
import json
from copy import deepcopy
from pathlib import Path

from Configs.loader import load_config
from simulation import ROOT, project_path, simulate
from reporting.run_report import build_run_tables
from reporting.history import capture_history
from diagnostics.errors import EvaluationFailure, format_failure


def write_history(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def write_events(history: list, path: Path) -> None:
    """Save substep events without downsampling them to the flight output rate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [event for state in history for event in state["plant"].fluids.events]
    fields = dict.fromkeys(("time_s", "kind", "component", "event", "count", "was_open", "is_open"))
    # Keep component-specific scalar transition data without a valve-type list.
    for event in events:
        fields.update((key, None) for key, value in event.items()
                      if isinstance(value, (str, int, float, bool)) or value is None)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=ROOT / "Configs" / "flight_pressure_fed_regulator.yaml"
    )
    parser.add_argument("--dt", type=float)
    parser.add_argument("--t-end", type=float)
    parser.add_argument("--sweep", type=Path, help="Run all cases from a sweep YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate a sweep without running flights")
    args = parser.parse_args()
    if args.sweep:
        from sensitivity import run_file
        run_file(args.sweep, dry_run=args.dry_run)
        return
    if args.dry_run:
        parser.error('--dry-run requires --sweep')
    cfg = deepcopy(load_config(args.config))
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

    try:
        result = simulate(cfg, record_history=True, compute_loads=True, progress=report_progress)
    except EvaluationFailure as error:
        output = cfg.get('simulation', {}).get('output')
        if output:
            path = project_path(output).with_suffix('.failure.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(error.diagnostic, indent=2))
        raise SystemExit(format_failure(error)) from None
    history = result.history
    if not history:
        raise RuntimeError(f"Flight simulation returned no time steps ({result.termination})")
    print(f"Propagation finished at t={history[-1]['kinematics'].t:.3f} s "
          f"in {time.perf_counter() - started:.1f} s. Writing history and plots...", flush=True)

    rows, summary = capture_history(result)
    output_path = project_path(cfg["simulation"]["output"])
    plot_path = project_path(cfg["simulation"]["plot"])
    write_history(rows, output_path)
    events_path = output_path.with_name(output_path.stem + "_events.csv")
    write_events(history, events_path)
    from reporting.flight_plots import plot_flight
    plots = plot_flight(history, rows, plot_path, summary.get("flight.burnout_time_s"))
    print(f"Run complete: {time.perf_counter() - setup_started:.1f} s total.", flush=True)

    print(f"Max altitude reached: {result.max_altitude:.1f} m")
    print(f"Apogee: {result.apogee:.1f} m" if result.apogee_reached else "Apogee: not reached")
    for table in build_run_tables(cfg, result, summary):
        print(table, end="\n\n")
    print(f"History: {output_path}")
    print(f"Events: {events_path}")
    for name, path in plots.items():
        print(f"{name.replace('_', ' ').title()}: {path}")


if __name__ == "__main__":
    main()
