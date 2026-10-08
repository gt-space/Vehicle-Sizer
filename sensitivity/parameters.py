"""Validated sweep declarations and isolated configuration overrides (SI units only)."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import product
from math import isfinite, prod
from pathlib import Path
from typing import Any

import yaml

from Configs.loader import load_config
ROOT = Path(__file__).resolve().parents[1]

# Public names are stable; use an existing dotted config path for additional fields.
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


@dataclass(frozen=True)
class SweepParameter:
    name: str
    path: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class SweepCase:
    case_id: str
    overrides: dict[str, float]
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
    """Explicit values, inclusive arithmetic interval, or linearly spaced interval."""
    if not isinstance(definition, dict):
        raise ValueError(f'{name} must be a mapping')
    if 'unit' in definition or 'units' in definition:
        raise ValueError(f'{name}: no unit conversions are supported; use base config SI values')
    specified = set(definition) - {'enabled'}
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
        start = _decimal(definition['start'], f'{name}.start')
        stop = _decimal(definition['stop'], f'{name}.stop')
        if 'step' in specified:
            step = _decimal(definition['step'], f'{name}.step')
            if not step or (stop - start) * step < 0:
                raise ValueError(f'{name}.step must advance toward stop')
            count = int((stop - start) / step) + 1
            if count < 1 or count > MAX_VALUES:
                raise ValueError(f'{name}: invalid or excessive number of sweep points')
            output = tuple(float(start + step * index) for index in range(count))
        else:
            count = definition['num']
            if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= MAX_VALUES:
                raise ValueError(f'{name}.num must be an integer >= 2')
            output = tuple(float(start + (stop - start) * index / (count - 1))
                           for index in range(count))
    if len(set(output)) != len(output):
        raise ValueError(f'{name}: duplicated parameter values')
    return output


def resolve_pressurant_tank(cfg: dict) -> str:
    # Infer from actual tank entries rather than assuming press_tank across all configs.
    matches = [name for name, tank in cfg.get('tanks', {}).items()
               if tank.get('type') == 'pressurant']
    if len(matches) != 1:
        raise ValueError('COPV sweep needs exactly one pressurant tank; use tanks.<id>.<field> instead')
    return matches[0]


def _get(cfg: dict, path: str) -> Any:
    node = cfg
    for part in path.split('.'):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f'Unknown configuration path: {path}')
        node = node[part]
    return node


def _set(cfg: dict, path: str, value: float) -> None:
    node = cfg
    for part in path.split('.')[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f'Unknown configuration path: {path}')
        node = node[part]
    if not isinstance(node, dict) or path.split('.')[-1] not in node:
        raise ValueError(f'Unknown configuration path: {path}')
    node[path.split('.')[-1]] = value


def _resolve_path(cfg: dict, name: str) -> str:
    path = PARAMETER_PATHS.get(name, name)
    if '{pressurant}' in path:
        path = path.replace('{pressurant}', resolve_pressurant_tank(cfg))
    # Target thrust and throat area are virtual paths on configs using the other mode.
    if name not in ('target_thrust', 'throat_area', 'thrust_tilt'):
        value = _get(cfg, path)
        _number(value, f'{name} (existing value)')
    elif name == 'thrust_tilt':
        if 'engine' not in cfg:
            raise ValueError('Thrust tilt requires an engine config')
    elif name == 'throat_area' and 'throat_area' not in cfg.get('engine', {}):
        if 'thrust_target' not in cfg.get('prop_system', {}):
            raise ValueError('Throat area requires an engine config')
    elif name == 'target_thrust' and 'thrust_target' not in cfg.get('prop_system', {}):
        if 'throat_area' not in cfg.get('engine', {}):
            raise ValueError('Target thrust requires an existing engine sizing input')
    return path


def _read_template(cfg: dict) -> dict:
    source = cfg.get('prop_system', {}).get('template')
    if isinstance(source, dict):
        return source
    if not isinstance(source, str):
        raise ValueError('Missing prop_system.template')
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    with path.open() as stream:
        return yaml.safe_load(stream)


def _template_references(template: Any) -> set[str]:
    if isinstance(template, dict):
        found = {template['config']} if set(template) == {'config'} else set()
        for value in template.values():
            found.update(_template_references(value))
        return found
    if isinstance(template, list):
        found = set()
        for value in template:
            found.update(_template_references(value))
        return found
    return set()


def parse_parameters(cfg: dict, sweep_cfg: dict) -> list[SweepParameter]:
    definition = sweep_cfg.get('sweep', {})
    raw = definition.get('parameters')
    if not isinstance(raw, dict) or not raw:
        raise ValueError('sweep.parameters must contain parameter declarations')
    params = []
    references = None
    for name, item in raw.items():
        if not isinstance(item, dict) or type(item.get('enabled')) is not bool:
            raise ValueError(f'{name}: enabled must be explicitly true or false')
        if not item['enabled']:
            continue
        path = _resolve_path(cfg, name)
        values = sweep_values(item, name)
        if name in {'chamber_pressure', 'copv_volume', 'copv_pressure', 'throat_area',
                    'target_thrust', 'weld_allowable'} and any(v <= 0 for v in values):
            raise ValueError(f'{name}: values must be positive')
        if name in {'cstar_efficiency', 'cf_efficiency'} and any(not 0 < v <= 1 for v in values):
            raise ValueError(f'{name}: efficiency must be in (0, 1]')
        if name in {'ox_injector_stiffness', 'fuel_injector_stiffness', 'ox_system_dp', 'fuel_system_dp'} and any(v < 0 for v in values):
            raise ValueError(f'{name}: values must be nonnegative')
        if path in ('prop_system.Pc_target', 'prop_system.ox_inj_stiffness',
                    'prop_system.fuel_inj_stiffness', 'prop_system.ox_tank_inj_dp',
                    'prop_system.fuel_tank_inj_dp'):
            if references is None:
                references = _template_references(_read_template(cfg))
            if path not in references:
                raise ValueError(f'{name}: {path} is not referenced by the selected propulsion template')
        params.append(SweepParameter(name, path, values))
    if not params:
        raise ValueError('Enable at least one sweep parameter')
    if {'throat_area', 'target_thrust'} <= {p.name for p in params}:
        raise ValueError('Enable throat_area OR target_thrust, not both')
    if len({p.path for p in params}) != len(params):
        raise ValueError('Multiple enabled parameters refer to the same configuration path')
    return params


def make_cases(cfg: dict, spec: dict) -> tuple[list[SweepParameter], list[SweepCase]]:
    parameters = parse_parameters(cfg, spec)
    mode = spec.get('sweep', {}).get('mode')
    if mode not in ('one_at_a_time', 'full_grid'):
        raise ValueError('sweep.mode must be one_at_a_time or full_grid')
    max_cases = spec.get('sweep', {}).get('max_cases', 10000)
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1:
        raise ValueError('sweep.max_cases must be a positive integer')
    count = (sum(len(p.values) for p in parameters) if mode == 'one_at_a_time'
             else prod(len(p.values) for p in parameters))
    if count + 1 > max_cases:
        raise ValueError(f'{count + 1} potential cases exceeds sweep.max_cases={max_cases}')
    # Deduplicate identical OAT baselines. In full grids the baseline is always separate,
    # and can later be excluded from surface/derivative calculations.
    cases = [SweepCase('baseline', {}, True)]
    seen = set()
    if mode == 'one_at_a_time':
        combinations = ({p.name: v} for p in parameters for v in p.values)
    else:
        combinations = (dict(zip((p.name for p in parameters), combo))
                        for combo in product(*(p.values for p in parameters)))
    for overrides in combinations:
        key = tuple(sorted(overrides.items()))
        if key in seen:
            continue
        seen.add(key)
        if mode == 'one_at_a_time':
            p = next(p for p in parameters if p.name in overrides)
            if p.name not in ('target_thrust', 'throat_area'):
                baseline = (cfg['engine'].get('thrust_tilt_deg', 0.0) if p.name == 'thrust_tilt'
                            else _get(cfg, p.path))
                if baseline == next(iter(overrides.values())):
                    continue
        cases.append(SweepCase(f'case_{len(cases):05d}', overrides))
    return parameters, cases


def apply_case(base_cfg: dict, case: SweepCase, parameters: list[SweepParameter]) -> dict:
    cfg = deepcopy(base_cfg)
    if case.is_baseline:
        return cfg
    paths = {p.name: p.path for p in parameters}
    for name, value in case.overrides.items():
        if name == 'target_thrust':
            cfg['prop_system']['thrust_target'] = value
            cfg['engine'].pop('throat_area', None)
            # Explicit design flows override mdot calculated from the new throat.
            cfg['prop_system'].pop('design_mdot_oxidizer', None)
            cfg['prop_system'].pop('design_mdot_fuel', None)
        elif name == 'thrust_tilt':
            cfg['engine']['thrust_tilt_deg'] = value
        elif name == 'throat_area':
            cfg['engine']['throat_area'] = value
            cfg['prop_system'].pop('thrust_target', None)
        else:
            _set(cfg, paths[name], value)
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
