"""Template-independent sweep parsing and candidate config overrides (native SI values).

Only genuinely exclusive sizing modes are treated specially. All ordinary
numeric inputs, including indexed section masses and inline template fields,
are resolved generically against the selected vehicle configuration.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import product
from math import isfinite, prod
from pathlib import Path
import re
from typing import Any

import yaml
from Configs.loader import load_config

ROOT = Path(__file__).resolve().parents[1]

# Retained for old sweep files. New sweeps should use the full path as the key.
PARAMETER_PATHS = {
    'chamber_pressure': 'prop_system.Pc_target',
    'ox_injector_stiffness': 'prop_system.ox_inj_stiffness',
    'fuel_injector_stiffness': 'prop_system.fuel_inj_stiffness',
    'ox_system_dp': 'prop_system.ox_tank_inj_dp',
    'fuel_system_dp': 'prop_system.fuel_tank_inj_dp',
    'copv_volume': 'tanks.{pressurant}.volume',
    'copv_pressure': 'tanks.{pressurant}.design_pressure',
    'throat_area': 'engine.throat_area',
    'target_thrust': 'prop_system.thrust_target',
    'cstar_efficiency': 'engine.cstar_efficiency',
    'cf_efficiency': 'engine.cf_efficiency',
    'thrust_tilt': 'engine.thrust_tilt_deg',
    'weld_allowable': 'advanced.weld_allowable',
}
MAX_VALUES = 100_000
_PATH_PART = re.compile(r'([^.\[\]]+)|\[(\d+)\]')


@dataclass(frozen=True)
class SweepParameter:
    name: str
    path: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class SweepCase:
    case_id: str
    overrides: dict[str, Any]
    is_baseline: bool = False


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f'{label} must be a finite number')
    result = float(value)
    if not isfinite(result):
        raise ValueError(f'{label} must be finite')
    return result


def _decimal(value: Any, label: str) -> Decimal:
    _number(value, label)
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f'Invalid {label}') from exc


def sweep_values(definition: dict, name: str) -> tuple[float, ...]:
    """Return numeric values from a list, arithmetic steps, or linspace."""
    if not isinstance(definition, dict):
        raise ValueError(f'{name} must be a mapping')
    if 'unit' in definition or 'units' in definition:
        raise ValueError(f'{name}: use native configuration units; no unit conversions')
    specified = set(definition) - {'enabled', 'path'}
    if 'values' in specified:
        if specified != {'values'}:
            raise ValueError(f'{name}: use values OR start/stop/step/num')
        values = definition['values']
        if not isinstance(values, list) or not values:
            raise ValueError(f'{name}.values must be a nonempty list')
        if len(values) > MAX_VALUES:
            raise ValueError(f'{name}: too many sweep points')
        output = tuple(_number(v, name) for v in values)
    else:
        if not {'start', 'stop'} <= specified or specified - {'start', 'stop', 'step', 'num'}:
            raise ValueError(f'{name}: provide values or start, stop, and step/num')
        if ('step' in specified) == ('num' in specified):
            raise ValueError(f'{name}: provide exactly one of step or num')
        start, stop = (_decimal(definition[key], f'{name}.{key}') for key in ('start', 'stop'))
        if 'step' in specified:
            step = _decimal(definition['step'], f'{name}.step')
            if not step or (stop - start) * step < 0:
                raise ValueError(f'{name}.step must advance toward stop')
            count = int((stop - start) / step) + 1
            if count < 1 or count > MAX_VALUES:
                raise ValueError(f'{name}: invalid number of sweep points')
            output = tuple(float(start + step * index) for index in range(count))
        else:
            count = definition['num']
            if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= MAX_VALUES:
                raise ValueError(f'{name}.num must be an integer >= 2')
            output = tuple(float(start + (stop - start) * i / (count - 1)) for i in range(count))
    if len(set(output)) != len(output):
        raise ValueError(f'{name}: duplicated values')
    return output


def resolve_pressurant_tank(cfg: dict) -> str:
    matches = [name for name, tank in cfg.get('tanks', {}).items()
               if isinstance(tank, dict) and tank.get('type') == 'pressurant']
    if len(matches) != 1:
        raise ValueError('COPV alias requires exactly one pressurant tank; use tanks.<id>.<field>')
    return matches[0]


def _parts(path: str) -> tuple[str | int, ...]:
    """Parse path segments, including numeric list selectors such as sections[2]."""
    if not isinstance(path, str) or not re.fullmatch(r'[A-Za-z_][\w-]*(?:\[\d+\])*(?:\.[A-Za-z_][\w-]*(?:\[\d+\])*)*', path):
        raise ValueError(f'Invalid configuration path: {path!r}')
    matches = list(_PATH_PART.finditer(path))
    if not matches:
        raise ValueError(f'Invalid configuration path: {path!r}')
    reconstructed = ''.join(m.group() if m.group().startswith('[') else m.group()
                            for m in matches)
    # Validate separators without allowing empty brackets or malformed expressions.
    if re.sub(r'\.', '', path) != reconstructed:
        raise ValueError(f'Invalid configuration path: {path!r}')
    return tuple(int(m.group(2)) if m.group(2) is not None else m.group(1) for m in matches)


def _get(cfg: dict, path: str) -> Any:
    node = cfg
    for key in _parts(path):
        if isinstance(key, int) and isinstance(node, list) and 0 <= key < len(node):
            node = node[key]
        elif isinstance(key, str) and isinstance(node, dict) and key in node:
            node = node[key]
        else:
            raise ValueError(f'Unknown configuration path: {path}')
    return node


def _set(cfg: dict, path: str, value: Any) -> None:
    segments = _parts(path)
    parent = cfg if len(segments) == 1 else _get(cfg, _path_string(segments[:-1]))
    key = segments[-1]
    if isinstance(key, int) and isinstance(parent, list) and 0 <= key < len(parent):
        parent[key] = value
    elif isinstance(key, str) and isinstance(parent, dict) and key in parent:
        parent[key] = value
    else:
        raise ValueError(f'Unknown configuration path: {path}')


def _path_string(parts: tuple[str | int, ...]) -> str:
    path = ''
    for key in parts:
        path += f'[{key}]' if isinstance(key, int) else ('.' if path else '') + key
    return path


def _read_template(cfg: dict) -> dict:
    selected = cfg.get('prop_system', {}).get('template')
    if isinstance(selected, dict):
        return deepcopy(selected)
    if not isinstance(selected, str):
        raise ValueError('Missing prop_system.template')
    path = Path(selected)
    if not path.is_absolute():
        path = ROOT / path
    with path.open() as stream:
        return yaml.safe_load(stream)


def _template_references(value: Any) -> set[str]:
    if isinstance(value, dict):
        found = {value['config']} if set(value) == {'config'} else set()
        for item in value.values():
            found.update(_template_references(item))
        return found
    if isinstance(value, list):
        found = set()
        for item in value:
            found.update(_template_references(item))
        return found
    return set()


def _get_candidate(cfg: dict, path: str) -> Any:
    if path.startswith('prop_system.template.') and isinstance(cfg.get('prop_system', {}).get('template'), str):
        template = _read_template(cfg)
        return _get(template, path[len('prop_system.template.'):])
    return _get(cfg, path)


def _resolve_path(cfg: dict, name: str, item: dict | None = None) -> str:
    path = (item or {}).get('path', PARAMETER_PATHS.get(name, name))
    if not isinstance(path, str):
        raise ValueError(f'{name}: path must be text')
    if '{pressurant}' in path:
        path = path.replace('{pressurant}', resolve_pressurant_tank(cfg))
    # Switching thrust design modes and adding a default zero tilt are safe
    # intentional exceptions to the existing-path-only rule.
    if path == 'engine.throat_area' and 'throat_area' not in cfg.get('engine', {}):
        if 'thrust_target' not in cfg.get('prop_system', {}):
            raise ValueError(f'{name}: engine needs throat_area or thrust_target')
    elif path == 'prop_system.thrust_target' and 'thrust_target' not in cfg.get('prop_system', {}):
        if 'throat_area' not in cfg.get('engine', {}):
            raise ValueError(f'{name}: engine needs throat_area or thrust_target')
    elif path == 'engine.thrust_tilt_deg' and 'thrust_tilt_deg' not in cfg.get('engine', {}):
        if 'engine' not in cfg:
            raise ValueError('Thrust tilt requires an engine config')
    else:
        original = _get_candidate(cfg, path)
        if isinstance(original, (dict, list)):
            raise ValueError(f'{name}: select a scalar value, not a mapping/list: {path}')
    return path


def _values_for(cfg: dict, name: str, path: str, item: dict) -> tuple[Any, ...]:
    try:
        original = _get_candidate(cfg, path)
    except ValueError:
        original = 0.0  # virtual throat/target/tilt
    if isinstance(original, bool):
        if set(item) - {'enabled', 'path', 'values'} or not isinstance(item.get('values'), list):
            raise ValueError(f'{name}: boolean sweeps require values: [true, false]')
        values = tuple(item['values'])
        if not values or any(type(v) is not bool for v in values):
            raise ValueError(f'{name}: boolean values required')
    elif isinstance(original, str):
        if set(item) - {'enabled', 'path', 'values'} or not isinstance(item.get('values'), list):
            raise ValueError(f'{name}: categorical sweeps require values: [...]')
        values = tuple(item['values'])
        if not values or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f'{name}: nonempty string values required')
    else:
        _number(original, f'{name} (baseline)')
        values = sweep_values(item, name)
        if isinstance(original, int) and not isinstance(original, bool):
            if any(not float(v).is_integer() for v in values):
                raise ValueError(f'{name}: integer configuration input needs whole-number values')
            values = tuple(int(v) for v in values)
    if len(set(values)) != len(values):
        raise ValueError(f'{name}: duplicated values')
    return values


def parse_parameters(cfg: dict, sweep_cfg: dict) -> list[SweepParameter]:
    raw = sweep_cfg.get('sweep', {}).get('parameters')
    if not isinstance(raw, dict) or not raw:
        raise ValueError('sweep.parameters must contain parameter declarations')
    params = []
    template_references = None
    for name, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError(f'{name}: parameter settings must be a mapping')
        enabled = item.get('enabled', True)
        if type(enabled) is not bool:
            raise ValueError(f'{name}: enabled must be true or false')
        if not enabled:
            continue
        path = _resolve_path(cfg, name, item)
        values = _values_for(cfg, name, path, item)
        # Explicit safety checks for preexisting aliases, not an attempt to
        # encode every possible thermodynamic constraint here.
        if (path in ('prop_system.Pc_target', 'engine.throat_area', 'prop_system.thrust_target')
                or (path.startswith('tanks.') and path.endswith(('.volume', '.design_pressure')))) and any(v <= 0 for v in values):
            raise ValueError(f'{name}: values must be positive')
        if path in ('engine.cstar_efficiency', 'engine.cf_efficiency') and any(not 0 < v <= 1 for v in values):
            raise ValueError(f'{name}: efficiencies must be in (0, 1]')
        # A referenced template path is validated for the current wiring. No
        # hardcoded 'pressure-fed only' template dispatch is necessary.
        if path in ('prop_system.Pc_target', 'prop_system.ox_inj_stiffness',
                    'prop_system.fuel_inj_stiffness', 'prop_system.ox_tank_inj_dp',
                    'prop_system.fuel_tank_inj_dp'):
            if template_references is None:
                template_references = _template_references(_read_template(cfg))
            if path not in template_references:
                raise ValueError(f'{name}: {path} is not referenced by the selected propulsion template')
        params.append(SweepParameter(name, path, values))
    if not params:
        raise ValueError('Specify at least one enabled sweep parameter')
    paths = [p.path for p in params]
    if len(set(paths)) != len(paths):
        raise ValueError('Multiple enabled parameters refer to the same configuration path')
    if {'engine.throat_area', 'prop_system.thrust_target'} <= set(paths):
        raise ValueError('Sweep engine.throat_area OR prop_system.thrust_target, not both')
    tokens = [_parts(path) for path in paths]
    for i, a in enumerate(tokens):
        if any(a == b[:len(a)] or b == a[:len(b)] for b in tokens[i+1:]):
            raise ValueError('Overlapping sweep configuration paths are not supported')
    return params


def make_cases(cfg: dict, spec: dict) -> tuple[list[SweepParameter], list[SweepCase]]:
    parameters = parse_parameters(cfg, spec)
    mode = spec.get('sweep', {}).get('mode', 'one_at_a_time')
    if mode not in ('one_at_a_time', 'full_grid'):
        raise ValueError('sweep.mode must be one_at_a_time or full_grid')
    max_cases = spec.get('sweep', {}).get('max_cases', 10000)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1:
        raise ValueError('sweep.max_cases must be a positive integer')
    count = (sum(len(p.values) for p in parameters) if mode == 'one_at_a_time'
             else prod(len(p.values) for p in parameters))
    if count + 1 > max_cases:
        raise ValueError(f'{count + 1} potential cases exceeds sweep.max_cases={max_cases}')
    cases = [SweepCase('baseline', {}, True)]
    seen = set()
    if mode == 'one_at_a_time':
        combinations = ({p.name: v} for p in parameters for v in p.values)
    else:
        combinations = (dict(zip((p.name for p in parameters), combo))
                        for combo in product(*(p.values for p in parameters)))
    by_name = {p.name: p for p in parameters}
    for overrides in combinations:
        key = tuple(sorted(overrides.items()))
        if key in seen:
            continue
        seen.add(key)
        if mode == 'one_at_a_time':
            name, value = next(iter(overrides.items()))
            path = by_name[name].path
            try:
                baseline = _get_candidate(cfg, path)
            except ValueError:
                baseline = 0.0  # virtual field
            if baseline == value:
                continue
        cases.append(SweepCase(f'case_{len(cases):05d}', overrides))
    return parameters, cases


def apply_case(base_cfg: dict, case: SweepCase, parameters: list[SweepParameter]) -> dict:
    cfg = deepcopy(base_cfg)
    if case.is_baseline:
        return cfg
    paths = {p.name: p.path for p in parameters}
    for name, value in case.overrides.items():
        path = paths[name]
        if path == 'prop_system.thrust_target':
            cfg['prop_system']['thrust_target'] = value
            cfg['engine'].pop('throat_area', None)
            cfg['prop_system'].pop('design_mdot_oxidizer', None)
            cfg['prop_system'].pop('design_mdot_fuel', None)
        elif path == 'engine.throat_area':
            cfg['engine']['throat_area'] = value
            cfg['prop_system'].pop('thrust_target', None)
        elif path == 'engine.thrust_tilt_deg':
            cfg['engine']['thrust_tilt_deg'] = value
        elif path.startswith('prop_system.template.'):
            if isinstance(cfg['prop_system']['template'], str):
                cfg['prop_system']['template'] = _read_template(cfg)
            _set(cfg, path, value)
        else:
            _set(cfg, path, value)
    return cfg


def load_sweep(path: str | Path) -> tuple[dict, dict, Path]:
    spec_path = Path(path).expanduser().resolve()
    with spec_path.open() as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict) or not isinstance(spec.get('base_config'), str):
        raise ValueError('Sweep YAML needs base_config: <vehicle YAML path>')
    source = Path(spec['base_config']).expanduser()
    if not source.is_absolute():
        project_relative = ROOT / source
        source = project_relative if project_relative.exists() else spec_path.parent / source
    return load_config(source), spec, source.resolve()
