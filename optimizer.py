"""Bounded launch-mass search. Physical units in; accepted designs alone can win."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from math import isfinite, pi, sqrt, nextafter
from pathlib import Path
from types import SimpleNamespace
import time
import traceback
from collections import Counter, deque
from errors import failure_details, SearchFailureLimit, LookupBoundsError

from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
import yaml

from AeroTables import DragModel
from AeroTables.dragmodel import _end as aero_endpoint
from Configs.loader import load_config
from constraints import ConstraintRecord, DesignInfeasible, EvaluationFailure, GeometryError, configured_limits, finalize
from Fluids.PropSystem import PropSystem
from Fluids.helpers.templates import load_template
from Fluids.design import initial_conditions
from Vehicle.Engine import Engine
from Vehicle.Vehicle import Vehicle
from simulation import project_path, property_sources, simulate
from simulation_types import SimResult
from constraints import vehicle_limit_margins

INCH = 0.0254
AERO_PATHS = {
    "omld": "vehicle.OMLD", "fineness": "nosecone.fineness_ratio",
    "boattail_aft": "fin_can.boattail_aft_diameter",
    "boattail_length": "fin_can.boattail_length", "span": "fin_can.span",
    "root": "fin_can.root_chord", "tip": "fin_can.tip_chord",
    "sweep_fraction": "fin_can.sweep_fraction", "thickness": "fin_can.fin_thickness",
}
DIMENSIONLESS = {"fineness", "sweep_fraction"}


def pump_roles(cfg):
    """Find optimizer pump variables from the same wiring used by PropSystem."""
    template = load_template(cfg, validate_pressures=False)
    roles = {}
    for branch in template['branches'].values():
        if branch.get('component', branch.get('model')) != 'pump':
            continue
        role = template['circuits'][branch['circuit']]['prop']
        if role not in ('oxidizer', 'fuel') or role in roles:
            raise ValueError('Optimizer supports at most one pump per propellant role')
        roles[role] = branch['pump_id']
    return roles


def variable_paths(cfg, tank_ids):
    paths = dict(AERO_PATHS)
    paths.update(chamber_pressure="prop_system.Pc_target", mixture_ratio="prop_system.MR_target",
                 thrust="prop_system.thrust_target", exit_pressure="engine.exit_pressure")
    pumps = pump_roles(cfg)
    for role in ("oxidizer", "fuel"):
        if role in pumps:
            pump = pumps[role]
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
        if isinstance(value, str):
            return endpoint(model.ranges[value][index], index)
        if isinstance(value, (list, tuple)):
            return float(value[0]) * endpoint(value[1], index)
        return float(value)
    return {name: tuple(endpoint(v, i) * (1 if name in DIMENSIONLESS else INCH)
                        for i, v in enumerate(model.ranges[name])) for name in AERO_PATHS}


def rejection(records):
    return SimResult(termination="infeasible_initial_design",
                     constraints={k: r.margin for k, r in records.items()}, constraint_records=records)


def domain_records(candidate, model):
    records = {}
    for name, limits in model.ranges.items():
        low, high = [aero_endpoint(v, candidate) for v in limits]
        factor = 1 if name in DIMENSIONLESS else INCH
        margin = min(candidate[name] - low, high - candidate[name])
        # Match DragModel.check's boundary tolerance after SI/inch conversion.
        if -1e-9 <= margin < 0:
            margin = 0.0
        records[f"aero.{name}"] = ConstraintRecord(
            margin * factor,
            "AeroTables", "sizing", "1" if name in DIMENSIONLESS else "m")
    return records


def pressurant_inner_radius(cfg, tank):
    """Match the selected vessel's wall model before checking its capacity."""
    diameter = float(tank["outer_diameter"])
    if tank["construction"] == "copv":
        wall = tank["thickness_slope"] * diameter + tank["thickness_intercept"]
    elif tank["construction"] == "metal":
        fos = tank.get("pressure_fos", cfg["advanced"]["tank_pressure_fos"])
        allowable = tank.get("weld_allowable", cfg["advanced"]["weld_allowable"])
        wall = fos * tank["design_pressure"] * diameter / (2 * allowable)
    else:
        raise ValueError(f"Unknown pressurant construction {tank['construction']!r}")
    radius = diameter / 2 - wall
    if not isfinite(radius) or radius <= 0 or tank["ellipse_ratio"] <= 1:
        raise ValueError("Invalid pressurant diameter, sized wall or ellipse ratio")
    return radius


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
    radius = pressurant_inner_radius(cfg, tank)
    head_volume = 4 * pi * radius**3 / (3 * tank["ellipse_ratio"])
    cylinder = ((tank["volume"] - head_volume) / (pi * radius**2)
                if tank.get("length") is None else tank["length"] - 2 * radius / tank["ellipse_ratio"])
    margins["geometry.copv_cylinder"] = (cylinder - 1e-9, "m")
    template = load_template(cfg, validate_pressures=False)
    for branch in template['branches'].values():
        if branch.get('component', branch.get('model')) == 'pump':
            role = template['circuits'][branch['circuit']]['prop']
            inlet = template['nodes'][branch['from']]['P0']
            margins[f'pump.{role}.inlet_pressure'] = (inlet - 1., 'Pa')
    return {k: ConstraintRecord(v, "optimizer", "sizing", unit) for k, (v, unit) in margins.items()}


def candidate_class(result):
    """Missing/unresolved physics must never outrank a fully assessed flight."""
    if result.accepted:
        return "feasible"
    if result.termination == "infeasible_initial_design":
        return "preflight_rejected"
    if (not result.completed or not result.burn_complete
            or not {"goal_apogee", "max_burn_duration"} <= result.constraint_records.keys()
            or any(r.required and r.margin is None for r in result.constraint_records.values())):
        return "unresolved"
    return "completed_infeasible"


def soft_penalties(result, settings):
    """Soft preferences never change feasibility; missing measurements cannot pass."""
    policy = settings.get('soft_penalties', {})
    if set(policy) - {'tank_pressure_tracking', 'max_q', 'stability_length_fraction'}:
        raise ValueError('Unknown soft penalty')
    penalties = {}
    for tank, spec in policy.get('tank_pressure_tracking', {}).items():
        for key in ('weight', 'pressure_scale_pa', 'time_scale_s'):
            value = spec[key]
            if not isfinite(value) or (value < 0 if key == 'weight' else value <= 0):
                raise ValueError(f'Invalid pressure penalty {key}')
        record = result.pressure_tracking.get(tank)
        if record is None or not isfinite(record['duration_s']) or record['duration_s'] <= 0:
            raise ValueError(f'Missing powered pressure tracking for {tank}')
        integral = record['integral_pa_s']
        if not isfinite(integral) or integral < 0:
            raise ValueError('Invalid pressure error integral')
        penalties[f'tank.{tank}.pressure_tracking'] = spec['weight'] * integral / (spec['pressure_scale_pa'] * spec['time_scale_s'])
    if 'max_q' in policy:
        spec = policy['max_q']
        if any(not isfinite(spec[k]) or spec[k] < 0 for k in ('weight', 'target_pa')) or not isfinite(spec['scale_pa']) or spec['scale_pa'] <= 0:
            raise ValueError('Invalid max-Q penalty')
        if not isfinite(result.max_q) or result.max_q < 0:
            raise ValueError('Invalid maximum dynamic pressure')
        penalties['max_q'] = spec['weight'] * max(0., result.max_q - spec['target_pa']) / spec['scale_pa']
    if 'stability_length_fraction' in policy:
        spec = policy['stability_length_fraction']
        lower, upper, scale, weight = (spec[k] for k in ('lower_target', 'upper_target', 'scale', 'weight'))
        if (not all(isfinite(v) for v in (lower, upper, scale, weight))
                or not 0 <= lower < upper or scale <= 0 or weight < 0):
            raise ValueError('Invalid stability length-fraction penalty')
        minimum, maximum = result.min_stability_length_fraction, result.max_stability_length_fraction
        if (minimum is None or maximum is None or not all(isfinite(v) for v in (minimum, maximum))
                or minimum > maximum):
            raise ValueError('Missing or invalid stability length-fraction extrema')
        penalties['stability_length_fraction'] = weight * (max(0., lower - minimum) + max(0., maximum - upper)) / scale
    return penalties


def candidate_score(result, settings, *, redundant=()):
    """Disjoint class intervals; scales guide search within a class only."""
    category = candidate_class(result)
    if category == "feasible":
        mass = result.initial_mass
        if mass is None or not isfinite(mass) or mass <= 0:
            raise ValueError("Accepted candidate has no finite positive launch mass")
        penalties = soft_penalties(result, settings)
        objective = mass / settings['reference_mass'] + sum(penalties.values())
        return min(objective / (1. + objective), nextafter(1., 0.)), {}
    contributions = {}
    for key, record in result.constraint_records.items():
        if not record.required or key in redundant or record.margin is None:
            continue
        scale = settings["constraint_scales"].get(key, settings["unit_scales"].get(record.units))
        if scale is None or not isfinite(scale) or scale <= 0:
            raise ValueError(f"Missing positive violation scale for {key} ({record.units})")
        if not isfinite(record.margin):
            raise ValueError(f"Nonfinite margin for {key}")
        if record.margin < 0:
            contributions[key] = -record.margin / scale
    if category == "unresolved":
        return 3., contributions
    offset = 2. if category == "preflight_rejected" else 1.
    violation = sum(contributions.values())
    score = offset + (1. - 1. / (1. + violation))
    return min(score, nextafter(offset + 1., offset)), contributions


class EvaluationBudget(Exception):
    pass


class FailureMonitor:
    """Candidate-ordered circuit breaker; physical/domain rejects are not solver failures."""
    def __init__(self, policy=None):
        policy = policy or {}
        self.consecutive_limit = policy.get('consecutive_limit', 5)
        self.window = policy.get('window', 50)
        self.rate_limit = policy.get('rate_limit', .2)
        self.repeated_limit = policy.get('repeated_limit', 3)
        for value in (self.consecutive_limit, self.window, self.repeated_limit):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Failure monitor limits must be positive integers')
        if not 0 < self.rate_limit <= 1:
            raise ValueError('Failure rate limit must be in (0, 1]')
        self.recent = deque(maxlen=self.window)
        self.consecutive = 0
        self.fingerprints = Counter()

    def observe(self, row):
        detail = row.get('failure_details') or {}
        kind = detail.get('kind', row.get('termination'))
        native_attempts = [attempt for attempt in row.get('worker_attempts', [])
                           if attempt['kind'] == 'native_crash']
        if native_attempts:
            kind = 'native_crash'
        failed = kind in {'solver_nonconvergence', 'residual_acceptance_failure',
                          'trial_domain_failure', 'native_crash', 'wall_timeout',
                          'worker_error', 'unexpected_error'}
        self.recent.append(failed)
        self.consecutive = self.consecutive + 1 if failed else 0
        if kind in {'unexpected_error', 'worker_error', 'native_crash'}:
            signature = detail.get('fingerprint', kind)
            if native_attempts:
                signature = f"native_crash:{native_attempts[-1]['exit_code']}"
            self.fingerprints[signature] += 1
            if self.fingerprints[signature] >= self.repeated_limit:
                raise SearchFailureLimit(f'Repeated {kind}: {signature}')
        if self.consecutive >= self.consecutive_limit:
            raise SearchFailureLimit(f'{self.consecutive} consecutive numerical/worker failures')
        if len(self.recent) == self.window and sum(self.recent) / self.window > self.rate_limit:
            raise SearchFailureLimit(f'Numerical/worker failure rate exceeds {self.rate_limit:.0%} in last {self.window} candidates')


class GenerationBudgetSolver(DifferentialEvolutionSolver):
    """Run to maxiter/evaluation budget; equal rejection scores are not convergence."""

    def converged(self):
        return False


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
        self.failure_monitor = FailureMonitor(settings.get('failure_policy'))

    def decode(self, values):
        if len(values) != len(self.paths):
            raise ValueError("Wrong design vector length")
        cfg = deepcopy(self.cfg)
        for value, (name, path), (low, high) in zip(values, self.paths.items(), self.bounds):
            if not isfinite(value) or not low <= value <= high:
                raise ValueError(f"Candidate {name} is outside configured bounds")
            set_path(cfg, path, float(value))
        if self.settings.get("conditional_geometry", False):
            return self.condition_geometry(cfg, values)
        return cfg

    def condition_geometry(self, cfg, values):
        """Map each box coordinate into its candidate-dependent physical interval.

        User bounds remain hard limits. Empty intersections are explicit design
        rejections, never clipping a derived nozzle or overriding a user bound.
        """
        coordinates = dict(zip(self.names, values))
        def assign(name, minimum=-float("inf"), maximum=float("inf")):
            lower, upper = self.settings["bounds"][name]
            lo, hi = max(lower, minimum), min(upper, maximum)
            if lo > hi:
                raise GeometryError({f"geometry.bounds.{name}": hi - lo})
            fraction = (coordinates[name] - lower) / (upper - lower) if upper > lower else 0.
            value = float(lo + fraction * (hi - lo))
            set_path(cfg, self.paths[name], value)
            return value

        clearance = float(cfg.get("geometry", {}).get("radial_clearance", 0.))
        fin, engine, prop = cfg["fin_can"], cfg["engine"], cfg["prop_system"]
        copv = cfg["tanks"][self.settings["tank_ids"]["pressurant"]]
        margin = 2 * (fin["boattail_wall_thickness"] + clearance) + 1e-9
        # The nozzle stays derived from the same CEA design calculation.
        epsilon = self.combustion.expansion_ratio(prop["Pc_target"], prop["MR_target"], engine["exit_pressure"])
        design = self.combustion.evaluate(chamber_pressure=prop["Pc_target"], mixture_ratio=prop["MR_target"],
            ambient_pressure=engine["exit_pressure"], expansion_ratio=epsilon,
            cstar_efficiency=engine["cstar_efficiency"], cf_efficiency=engine["cf_efficiency"])
        thrust_per_area = prop["Pc_target"] * design.Cf / epsilon
        exit_low = float(self.model.ranges["exit"][0]) * INCH
        aft_max = min(self.settings["bounds"]["boattail_aft"][1], self.settings["bounds"]["omld"][1])
        if aft_max <= margin:
            raise GeometryError({"engine.nozzle": aft_max - margin})
        thrust = assign("thrust", pi * exit_low**2 / 4 * thrust_per_area,
                        pi * (aft_max - margin)**2 / 4 * thrust_per_area)
        exit_diameter = sqrt(4 * thrust / thrust_per_area / pi)
        diameter = assign("omld", max(exit_diameter + margin,
            self.settings["bounds"]["boattail_aft"][0],
            copv["outer_diameter"] + 2 * (clearance) + 1e-9))
        # Span remains a physical length. Its permitted interval moves with D;
        # intersect it with the user's bounds before mapping the search coordinate.
        span_low, span_high = (aero_endpoint(bound, {"omld": diameter / INCH}) * INCH
                               for bound in self.model.ranges["span"])
        assign("span", span_low, span_high)
        length = assign("boattail_length", max(engine["length"], self.settings["bounds"]["root"][0]))
        aft_min = exit_diameter + margin
        if engine.get("envelope_diameter") is not None:
            envelope = engine["envelope_diameter"] + margin
            if diameter < envelope:
                raise GeometryError({"engine.envelope": diameter - envelope})
            aft_min = max(aft_min, diameter + (envelope - diameter) * length / engine["length"])
        assign("boattail_aft", aft_min, diameter)
        root = assign("root", maximum=length)
        assign("tip", maximum=root)
        radius = pressurant_inner_radius(cfg, copv)
        if copv.get("length") is None:
            assign("copv_volume", 4 * pi * radius**3 / (3 * copv["ellipse_ratio"]) + pi * radius**2 * 1e-8)

        if pump_roles(cfg):
            for role, leg in (("oxidizer", "ox"), ("fuel", "fuel")):
                outlet = prop["Pc_target"] * (1 + prop[f"{leg}_inj_stiffness"]) + prop[f"{leg}_inj_pumpout_dp"]
                assign(f"{role}_pump_head", maximum=outlet - 1.)
        # Match PropTank's pressure-sized wall and usable ellipsoidal-head volume.
        states = initial_conditions(cfg)
        for role in ("oxidizer", "fuel"):
            tank_id = self.settings["tank_ids"][role]
            tank, state = cfg["tanks"][tank_id], states[tank_id]
            pressure = Vehicle._max_tank_pressure(SimpleNamespace(cfg=cfg, initial_conditions=states), tank_id)
            wall = tank.get("wall_thickness")
            if wall is None:
                wall = (cfg["advanced"]["tank_pressure_fos"] * pressure * diameter
                        / (2 * cfg["advanced"]["weld_allowable"]))
            radius, tube = diameter / 2 - wall, tank["passthrough_diameter"] / 2
            if radius <= tube + clearance:
                raise GeometryError({f"{tank_id}.passthrough": radius - tube - clearance})
            head = 4 * pi / (3 * tank["ellipse_ratio"]) * (radius**2 - tube**2)**1.5
            rho = self.pure.state_pt(state["fluid"], state["P"], state["T"]).rho
            assign(f"{role}_mass", (head + pi * (radius**2 - tube**2) * 1e-8) * rho / tank["ullage_factor"])
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
            vehicle_limit_margins(cfg, vehicle)
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
        return set()

    def __call__(self, values):
        if self.count >= self.settings["max_evaluations"]:
            raise EvaluationBudget
        self.count += 1
        started = time.perf_counter()
        cfg, failure, details = None, None, None
        try:
            cfg = self.decode(values)
            result, redundant = self.evaluate(cfg)
        except GeometryError as error:
            result = rejection({k: ConstraintRecord(v, "optimizer", "sizing", "m") for k, v in error.constraints.items()})
            redundant = ()
        except (EvaluationFailure, LookupBoundsError) as error:
            phase = getattr(error, 'phase', 'sizing')
            details = failure_details(error, phase=phase)
            if details['fatal'] or (phase not in ("runtime", "flight initialization")
                                    and details['kind'] != 'table_domain_exceeded'):
                raise
            failure = traceback.format_exc()
            result = getattr(error, 'partial_result', None) or SimResult()
            result = deepcopy(result)
            result.apogee = None
            result.termination = details['kind']
            redundant = ()
            (self.output / f"failed_{self.count:04d}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            (self.output / f"failure_{self.count:04d}.txt").write_text(failure)
        except Exception as error:
            (self.output / "failed.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            (self.output / "failure.txt").write_text(traceback.format_exc())
            if isinstance(error, EvaluationFailure):
                raise
            raise EvaluationFailure("optimizer evaluation", cfg) from error
        score, contributions = candidate_score(result, self.settings, redundant=redundant)
        variables = ({name: get_path(cfg, path) for name, path in self.paths.items()} if cfg is not None else {})
        entry = dict(evaluation=self.count, variables=variables,
                     search_coordinates=list(map(float, values)), score_class=candidate_class(result), failure=failure,
                     failure_details=details,
                     pressure_tracking=result.pressure_tracking,
                     min_stability_length_fraction=result.min_stability_length_fraction,
                     max_stability_length_fraction=result.max_stability_length_fraction,
                     soft_penalties=soft_penalties(result, self.settings) if result.accepted else {},
                     score=score, accepted=result.accepted, mass=result.initial_mass,
                     termination=result.termination, warnings=result.warnings,
                     apogee=result.apogee, dry_mass=result.dry_mass,
                     burn_duration=result.burn_duration, violations=contributions,
                     constraints={k: asdict(v) for k, v in result.constraint_records.items()},
                     elapsed_seconds=time.perf_counter() - started)
        with (self.output / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        if result.accepted and (self.best is None or score < self.best["score"]):
            self.best = entry
            (self.output / "best.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            (self.output / "best.json").write_text(json.dumps(entry, indent=2, allow_nan=False))
        print(f"{self.count}: score={score:.6g}, accepted={result.accepted}, {result.termination}", flush=True)
        try:
            self.failure_monitor.observe(entry)
        except SearchFailureLimit as error:
            (self.output / 'stopped.json').write_text(json.dumps(dict(
                reason=str(error), evaluations=self.count, last_failure=details), indent=2))
            raise
        return score


def prepare(settings, model):
    settings = deepcopy(settings)
    cfg = load_config(project_path(settings["base_config"]))
    pumps = pump_roles(cfg)
    ids = settings["tank_ids"]
    if set(ids) != {"oxidizer", "fuel", "pressurant"} or len(set(ids.values())) != 3:
        raise ValueError("Optimizer tank roles must reference three distinct tanks")
    for role, tank_id in ids.items():
        expected = "pressurant" if role == "pressurant" else "propellant"
        if cfg["tanks"][tank_id]["type"] != expected:
            raise ValueError(f"{role} must reference a {expected} tank")
    if len(set(pumps.values())) != len(pumps):
        raise ValueError("Oxidizer and fuel must use distinct pumps")
    copv = cfg["tanks"][ids["pressurant"]]
    copv.pop("mass", None)
    if "volume_liters" in copv:
        copv["volume"] = copv.pop("volume_liters") * 1e-3
    states = cfg["prop_system"].get("initial_conditions", {})
    for role in ("oxidizer", "fuel"):
        states[ids[role]].pop("P", None)
    cfg.setdefault('constraints', {})
    for path, value in settings.get("evaluation_overrides", {}).items():
        set_path(cfg, path, value)
    limits = configured_limits(cfg)
    for key in settings.get('hard_constraints', []):
        if key in limits:
            continue
        parts = key.split('.')
        field = {'Tmin': 'min_temperature', 'Pmin': 'min_pressure'}.get(parts[-1])
        if len(parts) != 3 or parts[0] != 'tank' or field is None or field not in cfg.get('tanks', {}).get(parts[1], {}):
            raise ValueError(f'Hard constraint {key!r} requires a limit in its flight-config owner')
    tracking = settings.get('soft_penalties', {}).get('tank_pressure_tracking', {})
    for tank in tracking:
        if cfg.get('tanks', {}).get(tank, {}).get('type') != 'propellant':
            raise ValueError(f'Pressure tracking requires a propellant tank: {tank}')
    if 'max_q' in settings.get('soft_penalties', {}) and 'max_q' in limits:
        raise ValueError('max_q cannot be both a hard constraint and a soft preference in this policy')
    soft_penalties(SimpleNamespace(max_q=0., min_stability_length_fraction=0.,
        max_stability_length_fraction=0., pressure_tracking={
        tank: dict(duration_s=1., integral_pa_s=0.) for tank in tracking}), settings)
    if not {"goal_apogee", "max_burn_duration"} <= limits.keys():
        raise ValueError("Search requires goal_apogee and max_burn_duration")
    if cfg["environment"]["max_altitude"] <= limits["goal_apogee"]:
        raise ValueError("Atmosphere max_altitude must exceed goal_apogee")
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
    if settings['bounds']['exit_pressure'][1] >= settings['bounds']['chamber_pressure'][0]:
        raise ValueError('Exit-pressure upper bound must be below the chamber-pressure lower bound')
    combustion = cfg['property_models']['combustion']
    if combustion['source'] == 'table':
        from FluidTables.PropertyModels import TableCombustionPropertySource
        source = TableCombustionPropertySource(project_path(combustion['table']['lookup_file']),
                                               int(combustion['table']['nfz']))
        low, high = source.design_exit_pressure_bounds(settings['bounds']['chamber_pressure'],
                                                      settings['bounds']['mixture_ratio'])
        if not low <= settings['bounds']['exit_pressure'][0] <= settings['bounds']['exit_pressure'][1] <= high:
            raise ValueError(f'Exit-pressure bounds must lie in the common table sizing interval [{low}, {high}] Pa')
    if not isinstance(settings.get("conditional_geometry", False), bool):
        raise ValueError("conditional_geometry must be boolean")
    for name in ("reference_mass",):
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
                  score=candidate_score(result, evaluator.settings)[0],
                  pressure_tracking=result.pressure_tracking,
                  min_stability_length_fraction=result.min_stability_length_fraction,
                  max_stability_length_fraction=result.max_stability_length_fraction,
                  soft_penalties=soft_penalties(result, evaluator.settings) if result.accepted else {},
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
    x0 = initial if not settings.get("conditional_geometry") and all(lo <= x <= hi for x, (lo, hi) in zip(initial, evaluator.bounds)) else None
    try:
        with GenerationBudgetSolver(evaluator, evaluator.bounds, rng=settings["seed"],
                                    popsize=settings["popsize"], maxiter=settings["maxiter"],
                                    x0=x0, polish=False, workers=1) as solver:
            solver.solve()
        stop = "generation budget exhausted"
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
