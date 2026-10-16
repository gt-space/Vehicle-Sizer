"""Optimizer assessments only. Negative margins never throw simulation errors."""
from dataclasses import dataclass, replace
from math import isfinite


@dataclass(frozen=True)
class ConstraintRecord:
    margin: float | None
    source: str
    phase: str
    units: str
    required: bool = True
    time: float | None = None

    def __post_init__(self):
        if self.margin is not None and not isfinite(self.margin):
            raise ValueError(f"Nonfinite constraint margin: {self.margin}")
        if self.time is not None and not isfinite(self.time):
            raise ValueError(f"Nonfinite constraint time: {self.time}")


def merge_records(target, values, *, time=None):
    """Retain the worst measurement and its time together; records are immutable."""
    for key, record in values.items():
        previous = target.get(key)
        if (previous is None or record.margin is not None
                and (previous.margin is None or record.margin < previous.margin)):
            target[key] = (replace(record, time=time) if record.margin is not None
                           and record.time is None and time is not None else record)


def feasibility(records):
    required = [record.margin for record in records.values() if record.required]
    if any(value is not None and value < 0 for value in required):
        return False
    return None if not required or any(value is None for value in required) else True


UNITS = {"max_q": "Pa", "max_burn_duration": "s", "goal_apogee": "m",
         "min_stability_calibers": "calibers", "min_rail_twr": "1", "max_aoa_deg": "deg",
         "max_length_to_diameter": "1"}


def vehicle_limit_records(cfg, vehicle):
    """Assess static vehicle limits without changing simulation execution."""
    limit = cfg.get('constraints', {}).get('max_length_to_diameter')
    if limit is None:
        return {}
    length, diameter = float(vehicle.length), float(cfg['vehicle']['OMLD'])
    if not all(isfinite(v) and v > 0 for v in (length, diameter)):
        raise ValueError('L/D requires finite positive vehicle length and body diameter')
    return {'max_length_to_diameter': ConstraintRecord(float(limit) - length / diameter,
            'Vehicle.Vehicle', 'sizing', '1')}


class PropulsionAssessment:
    """Collect tank-limit margins on accepted substeps without stopping a run."""
    def __init__(self, propulsion):
        self.tank_limits = {}
        self.initial = dict(propulsion.sizing_constraints)
        for node_id, node in propulsion.node_definitions.items():
            tank_id = node.get('tank_id')
            if tank_id is None:
                continue
            definition = propulsion.tank_definitions.get(tank_id, {})
            limits = {name: float(definition[name]) for name in ('min_temperature', 'min_pressure')
                      if name in definition}
            if node.get('component') == 'pressurant_tank' and 'min_temperature' not in limits:
                values = [float(b['min_temperature']) for b in propulsion.branch_definitions.values()
                          if b['from'] == node_id and 'min_temperature' in b]
                if values:
                    limits['min_temperature'] = max(values)
            if any(not isfinite(v) or v <= 0 for v in limits.values()):
                raise ValueError('Tank constraint limits must be finite and positive')
            self.tank_limits[node_id] = (tank_id, limits)
            state = node['state0']
            if 'min_temperature' in limits:
                self.initial[f'tank.{tank_id}.Tmin'] = ConstraintRecord(
                    min(float(state[k]) for k in ('T', 'gas_T') if k in state) - limits['min_temperature'],
                    'Fluids.PropSystem', 'runtime', 'K', time=0.)
            if 'min_pressure' in limits:
                self.initial[f'tank.{tank_id}.Pmin'] = ConstraintRecord(
                    float(state['P']) - limits['min_pressure'], 'Fluids.PropSystem', 'runtime', 'Pa', time=0.)

    def __call__(self, state):
        margins = {}
        for node_id, (tank_id, limits) in self.tank_limits.items():
            node = state.nodes[node_id]
            if 'min_temperature' in limits:
                temperatures = [float(fluid['T']) for fluid in node.get('fluids', {}).values() if 'T' in fluid]
                if 'T' in node:
                    temperatures.append(float(node['T']))
                if not temperatures:
                    raise ValueError(f'Tank {tank_id!r} has no temperature to assess')
                margins[f'tank.{tank_id}.Tmin'] = min(temperatures) - limits['min_temperature']
            if 'min_pressure' in limits:
                margins[f'tank.{tank_id}.Pmin'] = float(node['P']) - limits['min_pressure']
        return {key: replace(self.initial[key], margin=value, time=None) for key, value in margins.items()}


def prepare_assessment(cfg, vehicle, propulsion):
    """Opt-in policy setup, after the shared physical build, for optimizers only."""
    from Vehicle.geometry_constraints import geometry_constraints
    limits = configured_limits(cfg)
    monitor = PropulsionAssessment(propulsion)
    propulsion.constraint_monitor = monitor
    merge_records(propulsion.network.constraint_records, monitor.initial)
    merge_records(propulsion.network.constraint_records, geometry_constraints(vehicle))
    merge_records(propulsion.network.constraint_records, vehicle_limit_records(cfg, vehicle))
    return limits


def configured_limits(cfg):
    limits = {key: float(value) for key, value in cfg.get("constraints", {}).items()}
    unknown = limits.keys() - UNITS.keys()
    if unknown:
        raise ValueError(f"Unknown simulation constraints: {sorted(unknown)}")
    if any(not isfinite(value) or value < 0 for value in limits.values()):
        raise ValueError("Constraint limits must be finite and nonnegative")
    if limits.get("max_aoa_deg", 15.0) > 15.0:
        raise ValueError("max_aoa_deg cannot exceed the 15 degree flight limit")
    return limits


def finalize(result, limits):
    """Attach optimizer assessment records without changing execution status."""
    sizing_only = result.termination == "infeasible_initial_design"
    for key, limit in limits.items():
        margin = None
        if key == 'max_length_to_diameter':
            if key in result.constraint_records:
                continue
        elif not sizing_only:
            if key == "goal_apogee":
                margin = None if result.apogee is None else result.apogee - limit
            elif key == "max_burn_duration":
                if result.burn_complete or result.burn_duration > limit:
                    margin = limit - result.burn_duration
            elif key == "max_q":
                margin = limit - result.max_q
            elif key == "max_aoa_deg":
                margin = limit - result.max_aoa_deg
            else:
                value = getattr(result, key)
                margin = None if value is None else value - limit
        source, phase = ('Vehicle.Vehicle', 'sizing') if key == 'max_length_to_diameter' else (
            'Flight.Flight', 'mission' if key in {'goal_apogee', 'max_burn_duration'} else 'runtime')
        result.constraint_records[key] = ConstraintRecord(margin, source, phase, UNITS[key],
            time=result.min_rail_twr_time if key == 'min_rail_twr' else None)
    return result
