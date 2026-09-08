from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Propulsion trace is empty: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows])
        for name in rows[0]
    }


def plot(input_path: Path, output_path: Path) -> None:
    data = load_trace(input_path)
    time = data["time_s"]
    newtons_per_lbf = 4.4482216152605

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(14, 12),
        sharex=True,
        constrained_layout=True,
    )

    thrust_axis = axes[0, 0]
    thrust_axis.plot(time, data["thrust_N"] / newtons_per_lbf)
    thrust_axis.axhline(2500.0, color="black", linestyle="--", label="design")
    thrust_axis.set_ylabel("Thrust [lbf]")
    thrust_axis.legend()

    pressure_axis = axes[0, 1]
    pressure_columns = {
        "Pc_Pa": "chamber",
        "ox_ullage_P_Pa": "oxidizer ullage",
        "ox_inj_in_P_Pa": "oxidizer injector inlet",
        "fuel_ullage_P_Pa": "fuel ullage",
        "fuel_inj_in_P_Pa": "fuel injector inlet",
    }
    for name, label in pressure_columns.items():
        pressure_axis.plot(time, data[name] / 1.0e6, label=label)
    pressure_axis.set_ylabel("Propellant-system pressure [MPa]")
    pressure_axis.legend(fontsize=8, ncol=2)

    press_axis = pressure_axis.twinx()
    press_axis.plot(
        time,
        data["press_tank_P_Pa"] / 1.0e6,
        color="black",
        linestyle="--",
        label="COPV",
    )
    press_axis.set_ylabel("COPV pressure [MPa]")
    press_axis.legend(loc="lower right", fontsize=8)

    mr_axis = axes[1, 0]
    mr_axis.plot(time, data["MR"])
    mr_axis.axhline(2.0, color="black", linestyle="--", label="design")
    mr_axis.set_ylabel("Mixture ratio")
    mr_axis.legend()

    flow_axis = axes[1, 1]
    flow_axis.plot(time, data["mdot_ox_kg_s"], label="oxidizer")
    flow_axis.plot(time, data["mdot_fuel_kg_s"], label="fuel")
    flow_axis.plot(time, data["mdot_nozzle_kg_s"], label="nozzle total")
    flow_axis.set_ylabel("Mass flow [kg/s]")
    flow_axis.legend()

    mass_axis = axes[2, 0]
    mass_axis.plot(time, data["press_tank_mass_kg"], label="COPV")
    mass_axis.plot(time, data["ox_ullage_mass_kg"], label="oxidizer tank")
    mass_axis.plot(time, data["fuel_ullage_mass_kg"], label="fuel tank")
    mass_axis.set_ylabel("Stored fluid mass [kg]")
    mass_axis.set_xlabel("Time [s]")
    mass_axis.legend()

    balance_axis = axes[2, 1]
    balance_axis.plot(time, data["mass_balance_error_kg"])
    balance_axis.axhline(0.0, color="black", linewidth=0.8)
    balance_axis.set_ylabel("Mass-balance error [kg/step]")
    balance_axis.set_xlabel("Time [s]")
    balance_axis.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))

    for axis in axes.flat:
        axis.grid(alpha=0.3)

    figure.suptitle("Three-Minute Propulsion Simulation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=ROOT / "outputs/propulsion_3min_harness.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/propulsion_3min_harness.png",
    )
    arguments = parser.parse_args()
    plot(arguments.input, arguments.output)
    print(f"Saved {arguments.output}")
