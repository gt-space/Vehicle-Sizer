"""Bounded launch-mass search. Physical units in; accepted designs alone can win."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from math import isfinite, pi
from pathlib import Path
import time
import traceback

from scipy.optimize import differential_evolution
import yaml

from AeroTables import DragModel
from Configs.loader import load_config
from constraints import ConstraintRecord, DesignInfeasible, EvaluationFailure, GeometryError, configured_limits, finalize
from Fluids.PropSystem import PropSystem
from Vehicle.Engine import Engine
from Vehicle.Vehicle import Vehicle
from simulation import project_path, property_sources, simulate
from simulation_types import SimResult

INCH = 0.0254
AERO_PATHS = {
    "omld": "vehicle.OMLD", "fineness": "nosecone.fineness_ratio",
    "boattail_aft": "fin_can.boattail_aft_diameter",
    "boattail_length": "fin_can.boattail_length", "span": "fin_can.span",
    "root": "fin_can.root_chord", "tip": "fin_can.tip_chord",
    "sweep_fraction": "fin_can.sweep_fraction", "thickness": "fin_can.fin_thickness",
}
DIMENSIONLESS = {"fineness", "sweep_fraction"}


def variable_paths(cfg, tank_ids):
    paths = dict(AERO_PATHS)
    paths.update(chamber_pressure="prop_system.Pc_target", mixture_ratio="prop_system.MR_target",
                 thrust="prop_system.thrust_target")
    for role in ("oxidizer", "fuel"):
        pump = cfg["prop_system"]["pump_ids"][role]
        paths[f"{role}_pump_head"] = f"prop_system.pumps.{pump}.pressure_rise_pa"
        paths[f"{role}_mass"] = f"tanks.{tank_ids[role]}.propellant_mass"
    paths.update(copv_pressure=f"tanks.{tank_ids['pressurant']}.design_pressure",
                 copv_volume=f"tanks.{tank_ids['pressurant']}.volume")
    return paths


def get_path(cfg, path):
    for key in path.split("."):
        cfg = cfg[key]
    return cfg


def set_path(cfg, path, value):
    *parents, key = path.split(".")
    for parent in parents:
        cfg = cfg[parent]
    cfg[key] = value


def aero_bounds(model):
    """Independent box bounds in SI; coupled bounds remain candidate checks."""
    def endpoint(value, index):
        return endpoint(model.ranges[value][index], index) if isinstance(value, str) else float(value)
    return {name: tuple(endpoint(v, i) * (1 if name in DIMENSIONLESS else INCH)
                        for i, v in enumerate(model.ranges[name])) for name in AERO_PATHS}


def rejection(records):
    return SimResult(termination="infeasible_initial_design",
                     constraints={k: r.margin for k, r in records.items()}, constraint_records=records)


def domain_records(candidate, model):
    records = {}
    for name, limits in model.ranges.items():
        low, high = [candidate[v] if isinstance(v, str) else v for v in limits]
        factor = 1 if name in DIMENSIONLESS else INCH
        records[f"aero.{name}"] = ConstraintRecord(
            min(candidate[name] - low, high - candidate[name]) * factor,
            "AeroTables", "sizing", "1" if name in DIMENSIONLESS else "m")
    return records


def primitive_records(cfg, tank_ids):
    """Known coupled input restrictions; do not catch arbitrary ValueErrors."""
    fin, prop = cfg["fin_can"], cfg["prop_system"]
    margins = {
        "geometry.fin_tip": (fin["root_chord"] - fin["tip_chord"], "m"),
        "geometry.fin_root": (fin["boattail_length"] - fin["root_chord"], "m"),
        "engine.length": (fin["boattail_length"] - cfg["engine"]["length"], "m"),
        "geometry.boattail": (cfg["vehicle"]["OMLD"] - fin["boattail_aft_diameter"], "m"),
        "engine.exit_pressure": (prop["Pc_target"] - cfg["engine"]["exit_pressure"] - 1., "Pa"),
    }
    tank = cfg["tanks"][tank_ids["pressurant"]]
    radius = (tank["outer_diameter"] - 2 * tank["wall_thickness"]) / 2
    if radius <= 0 or tank["ellipse_ratio"] <= 1:
        raise ValueError("Invalid fixed COPV diameter, wall or ellipse ratio")
    head_volume = 4 * pi * radius**3 / (3 * tank["ellipse_ratio"])
    margins["geometry.copv_cylinder"] = ((tank["volume"] - head_volume) / (pi * radius**2) - 1e-9, "m")
    for role, leg in (("oxidizer", "ox"), ("fuel", "fuel")):
        pump = prop["pumps"][prop["pump_ids"][role]]
        inlet = prop["Pc_target"] * (1 + prop[f"{leg}_inj_stiffness"]) + prop[f"{leg}_inj_pumpout_dp"] - pump["pressure_rise_pa"]
        margins[f"pump.{role}.inlet_pressure"] = (inlet - 1., "Pa")
    return {k: ConstraintRecord(v, "optimizer", "sizing", unit) for k, (v, unit) in margins.items()}


def candidate_score(result, settings, *, redundant=()):
    """Scales change search guidance, never the acceptance thresholds."""
    if result.accepted:
        mass = result.initial_mass
        if mass is None or not isfinite(mass) or mass <= 0:
            raise ValueError("Accepted candidate has no finite positive launch mass")
        return mass / (mass + settings["reference_mass"]), {}
    contributions = {}
    preflight = result.termination == "infeasible_initial_design"
    if not preflight and not result.completed:
        contributions["completion"] = settings["completion_penalty"]
    for key, record in result.constraint_records.items():
        if not record.required or key in redundant:
            continue
        if record.margin is None:
            if preflight or (key == "goal_apogee" and not result.completed):
                continue
            contributions[key] = settings["missing_penalty"]
        else:
            scale = settings["constraint_scales"].get(key, settings["unit_scales"].get(record.units))
            if scale is None or not isfinite(scale) or scale <= 0:
                raise ValueError(f"Missing positive violation scale for {key} ({record.units})")
            if not isfinite(record.margin):
                raise ValueError(f"Nonfinite margin for {key}")
            if record.margin < 0:
                contributions[key] = -record.margin / scale
    if not preflight and not result.burn_complete and "max_burn_duration" not in contributions:
        contributions["burn_completion"] = settings["missing_penalty"]
    return 1 + sum(contributions.values()), contributions


class EvaluationBudget(Exception):
    pass


class Evaluator:
    def __init__(self, cfg, settings, model, pure, combustion, output):
        self.cfg, self.settings = deepcopy(cfg), settings
        self.model, self.pure, self.combustion = model, pure, combustion
        self.output = Path(output)
        self.paths = variable_paths(cfg, settings["tank_ids"])
        self.names = list(self.paths)
        self.bounds = [settings["bounds"][name] for name in self.names]
        self.count = 0
        self.best = None

    def decode(self, values):
        if len(values) != len(self.paths):
            raise ValueError("Wrong design vector length")
        cfg = deepcopy(self.cfg)
        for value, (name, path), (low, high) in zip(values, self.paths.items(), self.bounds):
            if not isfinite(value) or not low <= value <= high:
                raise ValueError(f"Candidate {name} is outside configured bounds")
            set_path(cfg, path, float(value))
        return cfg

    def evaluate(self, cfg):
        records = primitive_records(cfg, self.settings["tank_ids"])
        if any(r.margin < 0 for r in records.values()):
            return rejection(records), set()
        propulsion = None
        try:
            # ponytail: size twice for passing candidates; share a prepared-build API
            # only if profiling shows construction materially affects search time.
            vehicle = Vehicle(cfg, self.pure)
            propulsion = PropSystem(cfg, vehicle.tanks, fluid_properties=self.pure,
                                    combustion_properties=self.combustion)
            vehicle.build(Engine(cfg["engine"]["mass"], cfg["engine"]["length"], propulsion.exit_area))
        except DesignInfeasible as error:
            result = SimResult(termination="infeasible_initial_design", constraints=dict(error.constraints))
            if isinstance(error, GeometryError):
                result.geometry_constraints, result.constraints = result.constraints, {}
            return finalize(result, configured_limits(cfg)), self.redundant(propulsion, cfg)
        records = domain_records(vehicle.aero_candidate(), self.model)
        if any(r.margin < 0 for r in records.values()):
            return rejection(records), self.redundant(propulsion)
        result = simulate(cfg, pure_properties=self.pure, combustion_properties=self.combustion,
                          aero_model=self.model)
        if result.max_altitude > cfg["environment"]["max_altitude"]:
            raise ValueError("Flight exceeded atmosphere table; increase environment.max_altitude")
        return result, self.redundant(propulsion)

    @staticmethod
    def redundant(propulsion, cfg=None):
        # Tank Pmin already contains the worst outgoing choked margin. Keep
        # standalone branch checks whose source has no tank aggregate.
        if propulsion is None:
            # Initial constraint rejection happens before the constructor returns.
            # Resolve only known tank-fed paths from the declared wiring.
            template = cfg["prop_system"].get("template") if cfg else None
            if template is not None:
                return {f"branch.{key}.choked" for key, branch in template["branches"].items()
                        if template["nodes"][branch["from"]].get("tank_id") is not None}
            mode = cfg["prop_system"].get("pressurization", "bang_bang") if cfg else "blowdown"
            return {f"branch.{leg}_{mode.upper().replace('BANG_BANG', 'BANGBANG')}.choked"
                    for leg in ("OX", "FUEL")} if mode != "blowdown" else set()
        return {f"branch.{key}.choked" for key, branch in propulsion.choked_branches.items()
                if propulsion.node_definitions[branch["from"]].get("tank_id") is not None}

    def __call__(self, values):
        if self.count >= self.settings["max_evaluations"]:
            raise EvaluationBudget
        cfg = self.decode(values)
        self.count += 1
        started = time.perf_counter()
        try:
            result, redundant = self.evaluate(cfg)
            score, contributions = candidate_score(result, self.settings, redundant=redundant)
        except Exception as error:
            (self.output / "failed.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            (self.output / "failure.txt").write_text(traceback.format_exc())
            if isinstance(error, EvaluationFailure):
                raise
            raise EvaluationFailure("optimizer evaluation", cfg) from error
        entry = dict(evaluation=self.count, variables=dict(zip(self.names, map(float, values))),
                     score=score, accepted=result.accepted, mass=result.initial_mass,
                     termination=result.termination, apogee=result.apogee,
                     burn_duration=result.burn_duration, violations=contributions,
                     constraints={k: asdict(v) for k, v in result.constraint_records.items()},
                     elapsed_seconds=time.perf_counter() - started)
        with (self.output / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        if result.accepted and (self.best is None or result.initial_mass < self.best["mass"]):
            self.best = entry
            (self.output / "best.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            (self.output / "best.json").write_text(json.dumps(entry, indent=2, allow_nan=False))
        print(f"{self.count}: score={score:.6g}, accepted={result.accepted}, {result.termination}", flush=True)
        return score


def prepare(settings, model):
    settings = deepcopy(settings)
    cfg = load_config(project_path(settings["base_config"]))
    if cfg["prop_system"].get("feed_type") != "pump_fed":
        raise ValueError("This 18-variable search requires a pump-fed base config")
    ids = settings["tank_ids"]
    if set(ids) != {"oxidizer", "fuel", "pressurant"} or len(set(ids.values())) != 3:
        raise ValueError("Optimizer tank roles must reference three distinct tanks")
    for role, tank_id in ids.items():
        expected = "pressurant" if role == "pressurant" else "propellant"
        if cfg["tanks"][tank_id]["type"] != expected:
            raise ValueError(f"{role} must reference a {expected} tank")
    if len(set(cfg["prop_system"]["pump_ids"].values())) != 2:
        raise ValueError("Oxidizer and fuel must use distinct pumps")
    copv = cfg["tanks"][ids["pressurant"]]
    copv.pop("mass", None)
    if "volume_liters" in copv:
        copv["volume"] = copv.pop("volume_liters") * 1e-3
    states = cfg["prop_system"].get("initial_conditions", cfg["prop_system"].get("state0", {}))
    for role in ("oxidizer", "fuel"):
        states[ids[role]].pop("P", None)
    for path, value in settings.get("evaluation_overrides", {}).items():
        set_path(cfg, path, value)
    limits = configured_limits(cfg)
    if not {"goal_apogee", "max_burn_duration"} <= limits.keys():
        raise ValueError("Search requires goal_apogee and max_burn_duration")
    if cfg["environment"]["max_altitude"] <= limits["goal_apogee"]:
        raise ValueError("Atmosphere max_altitude must exceed goal_apogee")
    schedule = cfg["aero"]["aoa_schedule"]
    if schedule[0][0] > 0 or schedule[-1][0] < cfg["simulation"]["t_end"]:
        raise ValueError("AoA schedule must cover the evaluation horizon")
    paths = variable_paths(cfg, ids)
    if set(settings["bounds"]) != set(paths):
        raise ValueError(f"Bounds must contain exactly: {list(paths)}")
    deck = aero_bounds(model)
    for name, bounds in settings["bounds"].items():
        if len(bounds) != 2:
            raise ValueError(f"{name} bounds require [lower, upper]")
        low, high = map(float, bounds)
        if not all(map(isfinite, (low, high))) or low <= 0 or low > high:
            raise ValueError(f"Invalid bounds for {name}")
        if name in deck and (low < deck[name][0] - 1e-12 or high > deck[name][1] + 1e-12):
            raise ValueError(f"{name} bounds exceed aero deck {deck[name]}")
        get_path(cfg, paths[name])
        settings["bounds"][name] = (low, high)
    for name in ("reference_mass", "completion_penalty", "missing_penalty"):
        if not isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("max_evaluations", "maxiter", "popsize"):
        if isinstance(settings[name], bool) or int(settings[name]) != settings[name] or settings[name] < (0 if name == "maxiter" else 1):
            raise ValueError(f"{name} must be an integer in range")
    for scale in (*settings["constraint_scales"].values(), *settings["unit_scales"].values()):
        if not isfinite(scale) or scale <= 0:
            raise ValueError("Constraint scales must be finite and positive")
    factor = settings["verification_dt_factor"]
    if not isfinite(factor) or not 0 < factor < 1:
        raise ValueError("verification_dt_factor must be between zero and one")
    return cfg, settings


def verify_best(evaluator):
    """One additional flight, outside the search budget; never hide failure."""
    cfg = load_config(evaluator.output / "best.yaml")
    cfg["simulation"]["dt"] *= evaluator.settings["verification_dt_factor"]
    try:
        result = simulate(cfg, pure_properties=evaluator.pure,
                          combustion_properties=evaluator.combustion,
                          aero_model=evaluator.model, compute_loads=True)
        if result.max_altitude > cfg["environment"]["max_altitude"]:
            raise ValueError("Verification exceeded atmosphere table")
    except Exception as error:
        (evaluator.output / "failed.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (evaluator.output / "failure.txt").write_text(traceback.format_exc())
        if isinstance(error, EvaluationFailure):
            raise
        raise EvaluationFailure("optimizer verification", cfg) from error
    report = dict(accepted=result.accepted, dt=cfg["simulation"]["dt"],
                  initial_mass=result.initial_mass, apogee=result.apogee,
                  burn_duration=result.burn_duration, termination=result.termination,
                  load_peaks=result.load_peaks,
                  constraints={k: asdict(v) for k, v in result.constraint_records.items()})
    (evaluator.output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if result.accepted:
        (evaluator.output / "verified.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return result.accepted


def optimize(settings, output):
    base = load_config(project_path(settings["base_config"]))
    model = DragModel(project_path(base["aero"]["model"]))
    cfg, settings = prepare(settings, model)
    pure, combustion = property_sources(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "search.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    (output / "base.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    evaluator = Evaluator(cfg, settings, model, pure, combustion, output)
    initial = [float(get_path(cfg, path)) for path in evaluator.paths.values()]
    x0 = initial if all(lo <= x <= hi for x, (lo, hi) in zip(initial, evaluator.bounds)) else None
    try:
        result = differential_evolution(evaluator, evaluator.bounds, seed=settings["seed"],
                                        popsize=settings["popsize"], maxiter=settings["maxiter"],
                                        x0=x0, polish=False, workers=1)
        stop = str(result.message)
    except EvaluationBudget:
        stop = "evaluation budget exhausted"
    verified = verify_best(evaluator) if evaluator.best is not None else None
    summary = dict(evaluations=evaluator.count, stop=stop, accepted=verified is True,
                   search_accepted=evaluator.best is not None, verified=verified,
                   best_mass=None if evaluator.best is None else evaluator.best["mass"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="Configs/optimizer.yaml")
    parser.add_argument("--output", required=True, help="New directory for this search")
    parser.add_argument("--max-evaluations", type=int)
    args = parser.parse_args()
    settings = load_config(args.config)
    if args.max_evaluations is not None:
        settings["max_evaluations"] = args.max_evaluations
    summary = optimize(settings, args.output)
    print(json.dumps(summary, indent=2))
    if not summary["accepted"]:
        print("No accepted design survived verification." if summary["search_accepted"] else "No accepted design found.")


if __name__ == "__main__":
    main()
