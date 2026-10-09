"""Generic, best-effort numerical export of *exposed* simulation state.

The exporter follows the public SimResult fields and accepted flight snapshots;
it does not inspect private solver internals or fabricate unavailable quantities.
Large spatial arrays become meaningful extrema, while the original load profiles
can optionally be preserved separately in HDF5 by the sweep runner.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from math import isfinite
from numbers import Number
from typing import Any

import numpy as np

# Numeric value extraction uses finite values only. These exclusions prevent
# duplicated megabyte-sized station distributions in the ordinary CSV/H5 rows.
MAX_DEPTH = 12


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
    elif isinstance(value, dict):
        for name, child in value.items():
            if isinstance(name, (str, int)):
                flatten_numbers(child, f'{prefix}.{name}', out, depth=depth+1, arrays=arrays)
    elif is_dataclass(value) and not isinstance(value, type):
        for entry in fields(value):
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


def snapshot_row(state: dict) -> dict[str, float]:
    """Export all numeric quantities exposed in one accepted flight snapshot.

    Public field keys mirror actual runtime structure; no fixed list of tank,
    regulator, pump, or generator names is necessary. Spatial load distributions
    get time-sample extrema under ``state.loads.<quantity>``.
    """
    row: dict[str, float] = {}
    if not isinstance(state, dict):
        return row
    for key, obj in state.items():
        # Station-resolved mass distributions are not sensible time-series
        # scalars. Structural load profiles are handled separately.
        if key == 'mass_properties' and isinstance(obj, dict):
            obj = {k: v for k, v in obj.items() if not isinstance(v, np.ndarray)}
        flatten_numbers(obj, f'state.{key}', row, arrays=(key == 'loads'))
    # A frequently requested derived aerodynamic variable: calibers.
    try:
        aero = state['plant'].aero
        mass = state['mass_properties']
        cp = float(aero.cp)
        cg = float(mass['cg'])
        diameter = float(mass['diameter'])
        if diameter > 0 and all(isfinite(v) for v in (cp, cg, diameter)):
            row['static_stability_margin_calibers'] = (cp - cg) / diameter
    except (KeyError, TypeError, AttributeError, ValueError, ZeroDivisionError):
        pass
    return row


def _is_powered(state: dict) -> bool | None:
    if isinstance(state.get('engine_on'), (bool, np.bool_)):
        return bool(state['engine_on'])
    try:
        return state['plant'].fluids.propulsion.mode != 'shutdown'
    except (KeyError, AttributeError, TypeError):
        return None


def _time(state: dict) -> float | None:
    try:
        value = float(state['kinematics'].t)
        return value if isfinite(value) else None
    except (KeyError, AttributeError, TypeError, ValueError):
        return None


def summarized_history(result: Any) -> dict[str, float]:
    """Initial, burnout, extrema and final values for every exposed numeric field.

    The launch snapshot (t=0) is used when available; otherwise the earliest
    saved flight endpoint is labeled as ``first_recorded``, not ``initial``.
    The powered-to-unpowered transition defines burnout. Incomplete burns do
    not receive fabricated burnout values. History values are sampled extrema,
    not continuous-time guarantees.
    """
    initial = getattr(result, 'initial_state', None)
    history = getattr(result, 'history', None) or []
    states = ([initial] if isinstance(initial, dict) else []) + [s for s in history if isinstance(s, dict)]
    if not states:
        return {}
    rows = [snapshot_row(state) for state in states]
    times = [_time(state) for state in states]
    burnout = None
    for i in range(1, len(states)):
        if _is_powered(states[i-1]) is True and _is_powered(states[i]) is False:
            burnout = i
            break
    # The result might have an explicit burn duration but no transition in the
    # sampled states; do not guess the EOL snapshot from a nearby time.
    summary: dict[str, float] = {}
    if burnout is not None and times[burnout] is not None:
        summary['flight.burnout_time_s'] = times[burnout]
    names = sorted({name for row in rows for name in row})
    for name in names:
        readings = [(i, row[name]) for i, row in enumerate(rows) if name in row]
        if not readings:
            continue
        prefix = f'history.{name}'
        if isinstance(initial, dict) and name in rows[0]:
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
        if burnout is not None and name in rows[burnout]:
            summary[f'{prefix}.burnout'] = rows[burnout][name]
    # Stable convenience field names for common plot requests.
    shortcuts = {
        'static_stability_margin_calibers': 'stability.static_margin_calibers',
        'state.kinematics.h': 'flight.altitude_m',
        'state.kinematics.x': 'flight.downrange_m',
        'state.kinematics.vz': 'flight.vertical_velocity_m_s',
        'state.kinematics.m': 'flight.mass_kg',
        'state.mass_properties.cg': 'vehicle.cg_m',
        'state.mass_properties.Iyy': 'vehicle.pitch_inertia_kg_m2',
        'state.mass_properties.total_mass': 'vehicle.mass_kg',
        'state.atmosphere.Ma': 'aero.mach',
        'state.atmosphere.q': 'aero.dynamic_pressure_Pa',
        'state.plant.aero.D': 'aero.drag_N',
        'state.plant.aero.Cd': 'aero.Cd',
        'state.plant.fluids.propulsion.thrust': 'propulsion.thrust_N',
        'state.plant.fluids.propulsion.Pc': 'propulsion.chamber_pressure_Pa',
        'state.plant.fluids.propulsion.MR': 'propulsion.mixture_ratio',
    }
    for original, alias in shortcuts.items():
        for phase in ('initial', 'first_recorded', 'burnout', 'minimum',
                      'maximum', 'maximum_absolute', 'final'):
            key = f'history.{original}.{phase}'
            if key in summary:
                summary[f'{alias}.{phase}'] = summary[key]
    for phase in ('initial', 'burnout', 'final'):
        altitude = summary.get(f'flight.altitude_m.{phase}')
        if altitude is not None:
            summary[f'flight.{phase}_altitude_m'] = altitude
        cg = summary.get(f'vehicle.cg_m.{phase}')
        if cg is not None:
            summary[f'vehicle.{phase}_cg_m'] = cg
    if burnout is not None:
        kin = states[burnout].get('kinematics')
        try:
            vx = float(kin.vx)
            vz = float(kin.vz)
            if isfinite(vx) and isfinite(vz):
                summary['flight.burnout_speed_m_s'] = float(np.hypot(vx, vz))
        except (AttributeError, TypeError, ValueError):
            pass
    # 'Minimum' and 'maximum' static margin are sampled over the accepted
    # states. They are distinct from the pre-existing optimizer constraint.
    if burnout is not None:
        summary['flight.burnout_reached'] = 1.0
    else:
        summary['flight.burnout_reached'] = 0.0
    # Approximate total impulse with powered flight samples; the event split
    # time is retained, but the last powered step uses zero-order hold because
    # the right-hand shutdown thrust is already zero.
    thrusts = [row.get('state.plant.fluids.propulsion.thrust') for row in rows]
    powered = [_is_powered(state) for state in states]
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
    return summary


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


def field_category(name: str) -> str:
    """Group available fields for discovery; not a fixed output whitelist."""
    low = name.lower()
    if 'static_stability' in low or 'static_margin' in low or 'stability_caliber' in low:
        return 'Stability'
    if low.startswith(('constraints.', 'constraint_', 'geometry_constraints.')) or low in ('feasible', 'accepted'):
        return 'Constraints'
    if 'thermal' in low or 'heat_' in low or 'temperature' in low or '.t_liq' in low or '.t_ull' in low:
        if any(token in low for token in ('press_tank', 'pressurant', 'copv')):
            return 'COPV'
        return 'Thermal'
    if any(token in low for token in ('press_tank', 'pressurant', 'copv')):
        return 'COPV'
    if any(token in low for token in ('ox_tank', 'fuel_tank', 'tank.')):
        return 'Propellant tanks'
    if any(token in low for token in ('pump', 'battery', 'motor_', 'electrical')):
        return 'Electrical / pumps'
    if any(token in low for token in ('gg_', 'gas_generator', 'gasgenerator')):
        return 'Gas generator'
    if any(token in low for token in ('regulator', 'valve', 'pressure_tracking', 'transition', 'switch_count')):
        return 'Controls'
    if any(token in low for token in ('loads.', 'bending', 'shear', 'axial_load', 'load_peaks', 'stress')):
        return 'Structural'
    if any(token in low for token in ('aero.', 'mach', 'dynamic_pressure', 'drag_', 'cd', 'angle_of_attack', 'aoa')):
        return 'Aerodynamics'
    if any(token in low for token in ('cg_', '.cg', 'inertia', 'iyy', 'ixx')):
        return 'Mass properties'
    if low.startswith(('design_summary.', 'vehicle.')) or any(token in low for token in ('dry_mass', 'initial_mass', 'fineness_ratio')):
        return 'Vehicle sizing'
    if any(token in low for token in ('thrust', 'chamber_pressure', 'mixture_ratio', 'mdot', 'propulsion.', 'impulse')):
        return 'Propulsion'
    if any(token in low for token in ('flight.', 'apogee', 'velocity', 'altitude', 'final_x', 'final_v', 'burn_duration')):
        return 'Flight performance'
    return 'Other'
