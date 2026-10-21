"""Shared extraction of recorded flight samples for CSVs, sweeps and reports."""
from dataclasses import fields, is_dataclass
from math import isfinite
from numbers import Number
from types import SimpleNamespace
from typing import Any

import numpy as np

from simulation_types import FluidOut, history_samples

MAX_DEPTH = 12
NODE_FIELDS = {'P', 'T', 'T_liq', 'T_ull', 'm', 'm_liq', 'm_ull',
               'U', 'U_liq', 'U_ull', 'quality'}


def flatten_numbers(value: Any, prefix: str, out: dict[str, float], *,
                    depth: int = 0, arrays: bool = False) -> None:
    """Flatten dictionaries, dataclasses, objects and simple sequences of numbers.

    With arrays=True, spatial numerical arrays are summarized by minimum,
    maximum and maximum absolute magnitude, not flattened into hundreds of
    per-station summary columns.
    """
    if depth > MAX_DEPTH or not prefix:
        return
    if isinstance(value, (bool, np.bool_)):
        out[prefix] = float(value)
    elif isinstance(value, Number) and not isinstance(value, complex):
        try:
            v = float(value)
            if isfinite(v):
                out[prefix] = v
        except (TypeError, ValueError, OverflowError):
            pass
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            flatten_numbers(value.item(), prefix, out, depth=depth+1, arrays=arrays)
        elif arrays and np.issubdtype(value.dtype, np.number):
            finite = np.asarray(value, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                out[f'{prefix}.minimum'] = float(finite.min())
                out[f'{prefix}.maximum'] = float(finite.max())
                out[f'{prefix}.maximum_absolute'] = float(np.max(np.abs(finite)))
    elif isinstance(value, (dict, SimpleNamespace)):
        for name, child in (value if isinstance(value, dict) else vars(value)).items():
            if isinstance(name, (str, int)):
                flatten_numbers(child, f'{prefix}.{name}', out, depth=depth+1, arrays=arrays)
    elif is_dataclass(value) and not isinstance(value, type):
        for entry in fields(value):
            if isinstance(value, FluidOut) and entry.name == "samples":
                continue
            flatten_numbers(getattr(value, entry.name), f'{prefix}.{entry.name}',
                            out, depth=depth+1, arrays=arrays)
    elif isinstance(value, (list, tuple)):
        # Small numeric sequences (e.g. sizing result tuples) can be addressed
        # individually; arrays of geometry are summarized instead.
        if len(value) <= 64:
            for index, child in enumerate(value):
                flatten_numbers(child, f'{prefix}[{index}]', out,
                                depth=depth+1, arrays=arrays)
        elif arrays and all(isinstance(item, Number) for item in value):
            flatten_numbers(np.asarray(value), prefix, out, depth=depth+1, arrays=True)


def snapshot_row(state: dict) -> dict:
    """Flatten one snapshot, adding only quantities derived from its state."""
    row = {}
    for key, obj in state.items():
        if key == 'before_event':
            continue  # history_samples exposes the left side as its own sample.
        if key == 'mass_properties' and isinstance(obj, dict):
            obj = {k: v for k, v in obj.items() if not isinstance(v, np.ndarray)}
        flatten_numbers(obj, f'state.{key}', row, arrays=(key == 'loads'))
    if {'state.kinematics.vx', 'state.kinematics.vz'} <= row.keys():
        vx, vz = row['state.kinematics.vx'], row['state.kinematics.vz']
        row['derived.speed_m_s'] = float(np.hypot(vx, vz))
        row['derived.flight_path_angle_rad'] = (float(np.arctan2(vz, vx))
            if row['derived.speed_m_s'] > 1e-8 else row.get('state.kinematics.theta', np.nan))
    aero = getattr(state.get('plant'), 'aero', None)
    cp = getattr(aero, 'total_cp', getattr(aero, 'cp', float('nan')))
    if isfinite(cp):
        row['derived.total_cp_m'] = cp
    if {'derived.total_cp_m', 'state.mass_properties.cg', 'state.mass_properties.diameter'} <= row.keys() and row['state.mass_properties.diameter'] > 0:
        row['derived.static_stability_margin_calibers'] = (
            row['derived.total_cp_m'] - row['state.mass_properties.cg']) / row['state.mass_properties.diameter']
    if {'state.forces.thrust', 'state.plant.aero.A', 'state.plant.aero.N', 'state.mass_properties.total_mass'} <= row.keys() and row['state.mass_properties.total_mass'] > 0:
        row['derived.proper_acceleration_g'] = float(np.hypot(
            row.get('state.forces.thrust_axial', row['state.forces.thrust'])
                - row.get('state.forces.axial_aero', row['state.plant.aero.A']),
            row.get('state.forces.normal_aero', row['state.plant.aero.N'])
                + row.get('state.forces.thrust_normal', 0.))
            / row['state.mass_properties.total_mass'] / 9.80665)
    fluids = getattr(state.get('plant'), 'fluids', None)
    mode = getattr(getattr(fluids, 'propulsion', None), 'mode', None)
    if mode is not None:
        row['state.plant.fluids.propulsion.mode'] = mode
    for name in ('state.engine_on', 'state.on_rail'):
        if name in row:
            row[name] = bool(row[name])
    for branch_id, branch in getattr(fluids, 'branch', {}).items():
        if 'is_open' in branch or 'is_switched' in branch:
            row[f'derived.branch.{branch_id}.switches'] = sum(
                event['kind'] == 'branch' and event['component'] == branch_id
                and ('is_open' in event or 'is_switched' in event) for event in fluids.events)
    return row


def capture_history(result: Any, detail: str = 'standard') -> tuple[list[dict], dict]:
    """Return endpoint rows and event-aware summaries from one extraction.

    Launch and pre-event samples contribute to extrema and impulse. Burnout is
    the first powered-to-unpowered transition; a nearby endpoint is never guessed.
    Without a launch snapshot, the earliest value is labeled first_recorded.
    Standard/full select exported columns only; summaries use all numeric fields.
    """
    if detail not in ('standard', 'full'):
        raise ValueError('History detail must be standard or full')
    initial = getattr(result, 'initial_state', None)
    states = list(history_samples(result))
    if not states:
        return [], {}
    rows = [snapshot_row(state) for state in states]
    times = [row.get('state.kinematics.t') for row in rows]
    powered = [row.get('state.engine_on') for row in rows]
    burnout = next((i for i in range(1, len(rows))
                    if powered[i-1] is True and powered[i] is False), None)
    summary: dict[str, float] = {}
    if burnout is not None and times[burnout] is not None:
        summary['flight.burnout_time_s'] = times[burnout]
    names = sorted({name for row in rows for name in row
                    if isinstance(row[name], Number)})
    for name in names:
        readings = [(i, row[name]) for i, row in enumerate(rows) if name in row and isfinite(row[name])]
        if not readings:
            continue
        prefix = f'history.{name}'
        if isinstance(initial, dict) and readings[0][0] == 0:
            summary[f'{prefix}.initial'] = rows[0][name]
        else:
            summary[f'{prefix}.first_recorded'] = readings[0][1]
        summary[f'{prefix}.final'] = readings[-1][1]
        minimum = min(readings, key=lambda item: item[1])
        maximum = max(readings, key=lambda item: item[1])
        maximum_absolute = max(readings, key=lambda item: abs(item[1]))
        summary[f'{prefix}.minimum'] = minimum[1]
        summary[f'{prefix}.maximum'] = maximum[1]
        summary[f'{prefix}.maximum_absolute'] = abs(maximum_absolute[1])
        if times[minimum[0]] is not None:
            summary[f'{prefix}.minimum_time_s'] = times[minimum[0]]
        if times[maximum[0]] is not None:
            summary[f'{prefix}.maximum_time_s'] = times[maximum[0]]
        if burnout is not None and name in rows[burnout] and isfinite(rows[burnout][name]):
            summary[f'{prefix}.burnout'] = rows[burnout][name]
    summary['flight.burnout_reached'] = float(burnout is not None)
    if {'history.state.mass_properties.cg.initial', 'history.state.mass_properties.cg.burnout'} <= summary.keys():
        summary['vehicle.cg_shift_burnout_m'] = summary['history.state.mass_properties.cg.burnout'] - summary['history.state.mass_properties.cg.initial']
    rail_time = getattr(result, 'rail_exit_time', None)
    if rail_time is not None:
        rail = next((row for row in rows if row.get('state.kinematics.t', -np.inf) >= rail_time - 1e-6), {})
        if 'state.forces.twr' in rail:
            summary['flight.rail_exit_twr'] = rail['state.forces.twr']
    thrusts = [row.get('state.plant.fluids.propulsion.thrust') for row in rows]
    if all(t is not None for t in times) and all(t is not None for t in thrusts):
        impulse = 0.0
        for i in range(1, len(states)):
            dt = times[i] - times[i - 1]
            if dt <= 0:
                continue
            if powered[i-1] and powered[i]:
                impulse += 0.5 * (thrusts[i-1] + thrusts[i]) * dt
            elif powered[i-1] and powered[i] is False:
                impulse += thrusts[i-1] * dt
        summary['propulsion.total_impulse_sampled_Ns'] = impulse
    by_state = {id(state): row for state, row in zip(states, rows)}
    endpoints = [by_state[id(state)] for state in getattr(result, 'history', None) or ()]
    if detail == 'standard':
        endpoints = [{name: value for name, value in row.items()
                      if not name.startswith(('state.loads.', 'state.plant.fluids.', 'state.plant.thermal.'))
                      or name.startswith(('state.plant.fluids.propulsion.', 'state.plant.fluids.branch.',
                                          'state.plant.fluids.mdot.', 'state.plant.fluids.event_counts.'))
                      or (name.startswith('state.plant.fluids.node.') and name.rsplit('.', 1)[-1] in NODE_FIELDS
                          and name.removeprefix('state.plant.fluids.node.').rsplit('.', 1)[0]
                              in states[0]['plant'].fluids.node)} for row in endpoints]
    return endpoints, summary


def spatial_profiles(history: list[dict]) -> dict[str, np.ndarray]:
    """Return station-resolved numerical loads only when spatial mesh is fixed.

    Caller persists these datasets to HDF5. Nonuniform/missing meshes are
    omitted rather than regridded or filled with invented sample values.
    """
    if not history or not all(isinstance(state, dict) and isinstance(state.get('loads'), dict)
                              and 'station' in state['loads'] for state in history):
        return {}
    keys = set.intersection(*(set(state['loads']) for state in history))
    result = {}
    for name in sorted(keys):
        arrays = [np.asarray(state['loads'][name]) for state in history]
        if arrays[0].ndim != 1 or not np.issubdtype(arrays[0].dtype, np.number):
            continue
        if not all(item.shape == arrays[0].shape and np.issubdtype(item.dtype, np.number)
                   for item in arrays):
            continue
        result[name] = np.stack(arrays).astype(float)
    return result
