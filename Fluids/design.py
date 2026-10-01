"""Pure design calculations shared by tank sizing and propulsion construction."""
from copy import deepcopy
from math import isfinite
from .helpers.templates import load_template


def pump_definition(prop: dict, pump_id: str) -> dict:
    definition = prop["pumps"][pump_id]
    if definition.get("drive") != "electric":
        raise ValueError(f"Pump {pump_id!r} requires drive: electric")
    for key in ("pressure_rise_pa", "efficiency", "max_power_kw", "gas_CdA"):
        value = float(definition[key])
        if not isfinite(value) or value <= 0:
            raise ValueError(f"Pump {pump_id!r} {key} must be finite and positive")
    if float(definition["efficiency"]) > 1:
        raise ValueError(f"Pump {pump_id!r} efficiency must be <= 1")
    return definition


def size_electric_pump(definition: dict, mdot: float, rho: float) -> dict:
    """Design-point electrical draw; efficiency includes pump, motor and drive."""
    if any(not isfinite(v) or v <= 0 for v in (mdot, rho)):
        raise ValueError("Pump design mass flow and liquid density must be finite and positive")
    rise = float(definition["pressure_rise_pa"])
    flow = mdot / rho
    power = rise * flow / (1000 * float(definition["efficiency"]))
    if not isfinite(power):
        raise ValueError("Pump design power must be finite")
    return dict(design_mdot=mdot, inlet_density=rho, volume_flow_m3_s=flow,
                pressure_rise_pa=rise, required_power_kw=power,
                max_power_kw=float(definition["max_power_kw"]),
                power_margin_kw=float(definition["max_power_kw"]) - power)


def tank_design_pressure(cfg: dict, tank_id: str, *, fallback=None) -> float:
    """Use the selected template's pressure relationship, or an explicit value."""
    if cfg['prop_system'].get('template') is not None:
        template = load_template(cfg)
        matches = [node for node in template['nodes'].values() if node.get('tank_id') == tank_id]
        if len(matches) != 1 or 'P0' not in matches[0]:
            raise ValueError(f'Template must declare one design pressure for tank {tank_id!r}')
        return float(matches[0]['P0'])
    value = cfg.get('tanks', {}).get(tank_id, {}).get('design_pressure', fallback)
    if value is None or not isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f'Tank {tank_id!r} needs a template design pressure or an explicit design_pressure')
    return float(value)


def initial_conditions(cfg: dict) -> dict:
    """Resolve P/T inputs, never store derived inventories in the design config.

    Temperatures are mandatory. Propellant initial P can override its design
    pressure; pressurant initial P is exclusively the tank design variable.
    """
    prop = cfg["prop_system"]
    template = prop.get("template")
    if isinstance(template, str) or (isinstance(template, dict) and any(
            isinstance(node.get("P0"), dict) for node in template["nodes"].values())):
        template = load_template(cfg)
    states = deepcopy(prop.get("initial_conditions", {}))
    for tank_id, state in states.items():
        required = {"T", "gas_T"} if "gas_fluid" in state else {"T"}
        missing = required.difference(state)
        if missing:
            raise ValueError(f"Tank {tank_id!r} requires explicit temperatures: {sorted(missing)}")
        definition = cfg.get("tanks", {}).get(tank_id, {})
        pressurant = definition.get("type") == "pressurant"
        if template is not None:
            pressurant |= any(node.get("tank_id") == tank_id and
                              node.get("component", node.get("model")) == "pressurant_tank"
                              for node in template["nodes"].values())
        if pressurant:
            if "design_pressure" not in definition:
                raise ValueError(f"Pressurant tank {tank_id!r} requires tanks.{tank_id}.design_pressure")
            state["P"] = tank_design_pressure(cfg, tank_id)
        elif "P" not in state:
            state["P"] = tank_design_pressure(cfg, tank_id)
        for name in ("P", "T", "gas_T"):
            if name not in state:
                continue
            try:
                value = float(state[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Tank {tank_id!r} requires finite positive {name}") from exc
            if not isfinite(value) or value <= 0:
                raise ValueError(f"Tank {tank_id!r} requires finite positive {name}")
    return states
