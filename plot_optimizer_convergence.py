"""Plot saved optimizer results without importing or running the solver.

Examples:
    python plot_optimizer_convergence.py outputs/pressure_fed_search_02
    python plot_optimizer_convergence.py outputs/pressure_fed_search_01 --window 250

Writes PNG and SVG files. Candidate index is search order, not completion order.
Supports serial evaluations.jsonl and parallel candidates/*/result.json logs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np


CONSTRAINTS = {
    "goal_apogee": "Apogee",
    "min_stability_calibers": "Minimum stability",
    "tank.press_tank.Pmin": "COPV minimum pressure",
    "max_length_to_diameter": "Length / diameter",
}
COLORS = ["#2563eb", "#d97706", "#9333ea", "#dc2626"]


def read_jsonl(path):
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                warnings.warn(f"Skipping unreadable record {path}:{line_number}")
    return records


def finite(value):
    return isinstance(value, (int, float)) and np.isfinite(value)


def load_results(directory):
    """Merge published records by ID, avoiding duplicates after finalization."""
    by_index = {}

    def add(row):
        index = row.get("index", row.get("evaluation"))
        if not isinstance(index, int) or index < 1 or not finite(row.get("score")):
            raise ValueError("Result requires a positive candidate index and finite score")
        by_index[index] = dict(row, index=index)

    for row in read_jsonl(directory / "evaluations.jsonl"):
        add(row)
    for path in sorted((directory / "candidates").glob("*/result.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            warnings.warn(f"Skipping unpublished/incomplete result {path}: {error}")
            continue
        add(row)
    rows = [by_index[index] for index in sorted(by_index)]
    if not rows:
        raise ValueError(f"No completed candidate results found in {directory}")
    return rows


def violation_frequency(rows, key, window):
    """Rate among assessed required margins in the last `window` saved results.

    Missing/unassessed margins are excluded, never counted as passing.
    """
    assessed, failed = [], []
    for row in rows:
        record = row.get("constraints", {}).get(key, {})
        margin = record.get("margin")
        valid = record.get("required", False) and finite(margin)
        assessed.append(int(valid))
        failed.append(int(valid and margin < 0))
    counts = np.concatenate(([0], np.cumsum(assessed)))
    failures = np.concatenate(([0], np.cumsum(failed)))
    right = np.arange(1, len(rows) + 1)
    left = np.maximum(0, right - window)
    denominator = counts[right] - counts[left]
    return np.divide(failures[right] - failures[left], denominator,
                     out=np.full(len(rows), np.nan), where=denominator > 0)


def plot_convergence(directory, output, window=500):
    rows = load_results(directory)
    x = np.array([row["index"] for row in rows])
    scores = np.array([row["score"] for row in rows])
    accepted = [row for row in rows if row.get("accepted") is True]
    feasible_mass = [row for row in accepted if finite(row.get("mass"))]
    best = min(rows, key=lambda row: row["score"])
    generations = read_jsonl(directory / "generations.jsonl")

    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True,
                                 gridspec_kw={"height_ratios": [1.1, 1, 1]},
                                 layout="constrained")
        fig.suptitle(f"Optimizer convergence — {directory.name}\n"
                     f"{len(rows):,} saved evaluations · {len(accepted):,} feasible · "
                     f"best score {best['score']:.6f}", fontsize=15)
        score_ax, mass_ax, constraints_ax = axes
        score_ax.axhspan(0, 1, color="#16a34a", alpha=.08, label="Feasible score range")
        score_ax.axhline(1, color="#16a34a", linewidth=.8, linestyle=":")
        score_ax.step(x, np.minimum.accumulate(scores), where="post",
                      color=COLORS[0], linewidth=2, label="Best score so far")
        median_rows = [row for row in generations
                       if finite(row.get("median_score")) and finite(row.get("evaluations"))]
        if median_rows:
            score_ax.plot([row["evaluations"] for row in median_rows],
                          [row["median_score"] for row in median_rows],
                          color=COLORS[1], linewidth=1.6, label="Retained population median")
        else:
            score_ax.text(.98, .92, "Population median not logged for this run",
                          transform=score_ax.transAxes, ha="right", color="#64748b", fontsize=9)
        if accepted:
            first = accepted[0]
            for ax in axes:
                ax.axvline(first["index"], color="#16a34a", linestyle="--", linewidth=.9)
            score_ax.scatter(first["index"], first["score"], color="#16a34a", zorder=5,
                             label=f"First feasible: #{first['index']}")
        score_ax.set(ylabel="Objective score ↓", title="1  Best objective and population progress")
        score_ax.legend(loc="upper right", fontsize=9, ncol=2)

        if feasible_mass:
            fx = [row["index"] for row in feasible_mass]
            masses = [row["mass"] for row in feasible_mass]
            mass_ax.scatter(fx, masses, color="#64748b", s=17, alpha=.55,
                            label="Feasible candidate launch mass")
            mass_ax.step(fx + [int(x[-1])], list(np.minimum.accumulate(masses)) + [min(masses)],
                         where="post", color="#16a34a", linewidth=2,
                         label="Lowest feasible launch mass so far")
            winner = min(feasible_mass, key=lambda row: row["score"])
            mass_ax.scatter(winner["index"], winner["mass"], marker="*", s=180,
                            color=COLORS[1], edgecolors="white", linewidth=.6, zorder=5,
                            label=f"Best score: #{winner['index']} ({winner['mass']:.2f} kg)")
            mass_ax.legend(loc="upper left", fontsize=9)
        else:
            mass_ax.text(.5, .5, "No feasible candidates yet", ha="center", va="center",
                         transform=mass_ax.transAxes, color="#64748b")
        mass_ax.set(ylabel="Launch mass (kg)", title="2  Feasible designs: mass and best-scoring candidate")

        for (key, label), color in zip(CONSTRAINTS.items(), COLORS):
            constraints_ax.plot(x, violation_frequency(rows, key, window),
                                label=label, color=color, linewidth=1.3)
        constraints_ax.set(ylabel="Violation frequency", ylim=(-.03, 1.06),
                           xlabel="Candidate evaluation index (search order)",
                           title=f"3  Constraint failures among assessed margins — rolling {window} saved evaluations")
        constraints_ax.yaxis.set_major_formatter(PercentFormatter(1))
        constraints_ax.legend(loc="upper right", fontsize=9, ncol=2)
        for ax in axes:
            ax.grid(alpha=.17)
            ax.set_xlim(0, max(2, int(x[-1])))
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        fig.supxlabel("Search feasibility only; final verification is separate. Missing constraint measurements are excluded.\n"
                      "Scores include soft penalties; crossing below 1 changes feasibility class. Parallel completion order may differ.",
                      fontsize=9, color="#475569")
        output.parent.mkdir(parents=True, exist_ok=True)
        paths = [Path(str(output) + suffix) for suffix in (".png", ".svg")]
        for path in paths:
            fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("search", type=Path, help="Saved optimizer search directory")
    parser.add_argument("--output", type=Path, help="Output prefix, without extension (default: SEARCH/convergence)")
    parser.add_argument("--window", type=int, default=500, help="Constraint-rate window in saved evaluations")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    for path in plot_convergence(args.search, args.output or args.search / "convergence", args.window):
        print(path.resolve())


if __name__ == "__main__":
    main()
