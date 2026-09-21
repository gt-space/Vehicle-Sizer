"""Pure design calculations shared by tank sizing and propulsion construction."""
from copy import deepcopy
from math import isfinite


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


def pressure_ladder(prop: dict) -> dict[str, float]:
    ladder = {}
    feed = prop.get("feed_type", prop.get("press_model", "pressure_fed"))
    for leg in ("ox", "fuel"):
        injector = float(prop["Pc_target"]) * (1.0 + float(prop[f"{leg}_inj_stiffness"]))
        ladder[f"P{leg}_inj"] = injector
        if feed == "pump_fed":
            if any(key in prop for key in (f"{leg}_pump_head", f"{leg}_pump_gas_cda")):
                raise ValueError("Move legacy pump head/gas CdA inputs into prop_system.pumps")
            role = "oxidizer" if leg == "ox" else "fuel"
            pump = pump_definition(prop, prop["pump_ids"][role])
            outlet = injector + float(prop[f"{leg}_inj_pumpout_dp"])
            inlet = outlet - float(pump["pressure_rise_pa"])
            ladder[f"P{leg}_pump_outlet"] = outlet
            ladder[f"P{leg}_pump_inlet"] = inlet
            tank = inlet + float(prop[f"{leg}_pumpin_tank_dp"])
        else:
            tank = injector + float(prop[f"{leg}_tank_inj_dp"])
        ladder[f"P{leg}_tank"] = tank
    if any(not isfinite(p) or p <= 0 for p in ladder.values()):
        raise ValueError("Design pressure ladder requires finite positive absolute pressures")
    return ladder


def tank_design_pressure(cfg: dict, tank_id: str, *, fallback=None) -> float:
    """Resolve a tank's declared design pressure or template-assigned feed role."""
    tank = cfg.get("tanks", {}).get(tank_id, {})
    role = tank.get("role")
    template = cfg["prop_system"].get("template")
    if role is None and template is not None:
        roles = {c["prop"] for c in template["circuits"].values() if c.get("tank_id") == tank_id}
        if len(roles) == 1:
            role = roles.pop()
    if role is None and template is None:
        # Tank names are roles only in the built-in legacy templates.
        role = {"ox_tank": "oxidizer", "fuel_tank": "fuel"}.get(tank_id)
    if role in ("oxidizer", "fuel"):
        if "design_pressure" in tank:
            raise ValueError(f"Propellant tank {tank_id!r} design pressure comes from the pressure ladder; use initial P for an initialization override")
        leg = "ox" if role == "oxidizer" else "fuel"
        return pressure_ladder(cfg["prop_system"])[f"P{leg}_tank"]
    if "design_pressure" in tank:
        pressure = float(tank["design_pressure"])
        if not isfinite(pressure) or pressure <= 0:
            raise ValueError(f"Tank {tank_id!r} design_pressure must be finite and positive")
        return pressure
    if fallback is not None:
        return float(fallback)
    raise ValueError(f"Tank {tank_id!r} needs a design_pressure or an explicit initial P")


def initial_conditions(cfg: dict) -> dict:
    """Resolve P/T inputs, never store derived inventories in the design config.

    Temperatures are mandatory. Propellant initial P can override its design
    pressure; pressurant initial P is exclusively the tank design variable.
    """
    prop = cfg["prop_system"]
    states = deepcopy(prop.get("initial_conditions", prop.get("state0", {})))
    for tank_id, state in states.items():
        required = {"T", "gas_T"} if "gas_fluid" in state else {"T"}
        missing = required.difference(state)
        if missing:
            raise ValueError(f"Tank {tank_id!r} requires explicit temperatures: {sorted(missing)}")
        definition = cfg.get("tanks", {}).get(tank_id, {})
        template = prop.get("template")
        pressurant = definition.get("type") == "pressurant"
        if template is not None:
            pressurant |= any(node.get("tank_id") == tank_id and
                              node.get("component", node.get("model")) == "pressurant_tank"
                              for node in template["nodes"].values())
        elif "type" not in definition:
            pressurant |= tank_id == "press_tank"
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
