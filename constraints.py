"""Shared candidate policy. Signed margins >= 0 pass; missing is not passing."""
from dataclasses import dataclass
from math import isfinite


class DesignInfeasible(ValueError):
    """Expected sizing rejection, not a numerical or programming failure."""
    def __init__(self, constraints, pump_sizing=None):
        validate_margins(constraints)
        self.constraints = constraints
        self.pump_sizing = pump_sizing or {}
        super().__init__(f"Infeasible design: {constraints}")


class GeometryError(DesignInfeasible):
    """Well-defined geometry with negative packaging clearances."""


class OperatingInfeasible(DesignInfeasible):
    """Converged equations outside a physical operating limit; never committed."""
    def __init__(self, constraints, time):
        self.time = time
        super().__init__(constraints)


class EvaluationFailure(RuntimeError):
    """Fatal: callers must not convert this to an optimizer penalty."""
    def __init__(self, phase, config, partial_result=None):
        self.phase = phase
        self.config = config
        self.partial_result = partial_result
        super().__init__(f"Candidate evaluation failed during {phase}; see chained exception")


@dataclass
class ConstraintRecord:
    margin: float | None
    source: str
    phase: str
    units: str
    required: bool = True
    time: float | None = None


def validate_margins(margins):
    for key, value in margins.items():
        if value is not None and not isfinite(value):
            raise ValueError(f"Nonfinite constraint: {key}={value}")


def merge_margins(target, values, *, times=None, sample_times=None, time=None):
    """Retain worst assessed margin, including accepted fluid substeps."""
    validate_margins(values)
    for key, value in values.items():
        previous = target.get(key)
        if value is not None and (previous is None or value < previous):
            target[key] = value
            if times is not None:
                times[key] = (sample_times or {}).get(key, time)
        elif key not in target:
            target[key] = None


def feasibility(margins):
    validate_margins(margins)
    if any(value is not None and value < 0 for value in margins.values()):
        return False
    return None if any(value is None for value in margins.values()) else True


UNITS = {"max_q": "Pa", "max_burn_duration": "s", "goal_apogee": "m",
         "min_stability_calibers": "calibers", "min_rail_twr": "1"}


def configured_limits(cfg):
    limits = {key: float(value) for key, value in cfg.get("constraints", {}).items()}
    unknown = limits.keys() - UNITS.keys()
    if unknown:
        raise ValueError(f"Unknown simulation constraints: {sorted(unknown)}")
    validate_margins(limits)
    if any(value < 0 for value in limits.values()):
        raise ValueError("Constraint limits must be nonnegative")
    return limits


def finalize(result, limits):
    """One reporting/acceptance contract for both preflight rejects and flights."""
    validate_margins(result.geometry_constraints)
    sizing_only = result.termination == "infeasible_initial_design"
    for key, limit in limits.items():
        margin = None
        if not sizing_only:
            if key == "goal_apogee":
                margin = None if result.apogee is None else result.apogee - limit
            elif key == "max_burn_duration":
                if result.burn_complete or result.burn_duration > limit:
                    margin = limit - result.burn_duration
            elif key == "max_q":
                margin = limit - result.max_q
            else:
                value = getattr(result, key)
                margin = None if value is None else value - limit
        result.constraints[key] = margin
    validate_margins(result.constraints)
    records = {}
    for key, margin in result.geometry_constraints.items():
        records[key] = ConstraintRecord(margin, "Vehicle.geometry_constraints", "sizing", "m", margin is not None)
    for key, margin in result.constraints.items():
        phase = "mission" if key in {"goal_apogee", "max_burn_duration"} else (
            "sizing" if key not in limits and (sizing_only or key.startswith("pump.")) else "runtime")
        source = "Flight.Flight" if key in limits else "Fluids.PropSystem"
        units = UNITS.get(key, "kW" if key.endswith("max_power") else "K" if key.endswith("Tmin") else "Pa")
        records[key] = ConstraintRecord(margin, source, phase, units,
                                        time=result.min_rail_twr_time if key == "min_rail_twr" else result.constraint_times.get(key))
    result.constraint_records = records
    return result
