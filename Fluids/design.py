"""design calculations which are shared by tank sizing and propulsion construction."""
from copy import deepcopy
from math import isfinite
from scipy.optimize import brentq
from diagnostics.errors import LookupBoundsError
from .helpers.templates import bind_template


def pump_definition(prop: dict, pump_id: str) -> dict:
    definition = prop["pumps"][pump_id]
    if definition.get("drive") not in {"electric", "gas_generator"}:
        raise ValueError(f"Pump {pump_id!r} requires drive: electric or gas_generator")
    for key in ("pressure_rise_pa", "efficiency", "gas_CdA") + (("max_power_kw",) if definition["drive"] == "electric" else ()):
        value = float(definition[key])
        if not isfinite(value) or value <= 0:
            raise ValueError(f"Pump {pump_id!r} {key} must be finite and positive")
    if float(definition["efficiency"]) > 1:
        raise ValueError(f"Pump {pump_id!r} efficiency must be <= 1")
    return definition


def size_pump(definition: dict, mdot: float, rho: float) -> dict:
    """Shared design power; efficiency is pump-only for a turbine drive."""
    if any(not isfinite(v) or v <= 0 for v in (mdot, rho)):
        raise ValueError("Pump design mass flow and liquid density must be finite and positive")
    rise = float(definition["pressure_rise_pa"])
    flow = mdot / rho
    power = rise * flow / (1000 * float(definition["efficiency"]))
    if not isfinite(power):
        raise ValueError("Pump design power must be finite")
    result = dict(design_mdot=mdot, inlet_density=rho, volume_flow_m3_s=flow,
                  pressure_rise_pa=rise, hydraulic_power_w=rise * flow,
                  drive=definition["drive"])
    if result["drive"] == "electric":
        result.update(required_power_kw=power,
                      max_power_kw=float(definition["max_power_kw"]),
                      power_margin_kw=float(definition["max_power_kw"]) - power)
    else:
        result["required_shaft_power_w"] = 1000 * power
    return result


def size_gas_generator(power_w, inlet_pressure_pa, definition, properties):
    """Fixed design demand from constant-property isentropic turbine expansion."""
    outlet = float(definition["turbine_outlet_pressure_pa"])
    efficiency = float(definition["efficiency"])
    mr = float(definition["mixture_ratio"])
    temperature, gamma, gas_constant = (float(properties[k]) for k in ("T", "gamma", "R"))
    if (not all(isfinite(v) and v > 0 for v in
                (power_w, inlet_pressure_pa, outlet, efficiency, mr, temperature, gas_constant))
            or not isfinite(gamma) or gamma <= 1 or efficiency > 1 or outlet >= inlet_pressure_pa):
        raise ValueError("GG requires positive finite power/properties, gamma > 1, 0 < efficiency <= 1 and inlet > outlet pressure")
    # Temporary perfect-gas cp [J/(kg K)]; replace with tabulated chamber cp.
    cp = gamma * gas_constant / (gamma - 1)
    ratio = inlet_pressure_pa / outlet
    temperature_ratio = ratio ** (-(gamma - 1) / gamma)
    delta_t = temperature * (1 - temperature_ratio)
    work = efficiency * cp * delta_t
    if not all(isfinite(v) and v > 0 for v in (cp, work)):
        raise ValueError("GG design specific work must be finite and positive")
    mdot = power_w / work
    if not isfinite(mdot) or mdot <= 0:
        raise ValueError("GG design specific work and flow must be finite and positive")
    return dict(required_shaft_power_w=power_w, inlet_pressure_pa=inlet_pressure_pa,
                outlet_pressure_pa=outlet, efficiency=efficiency, mixture_ratio=mr,
                pressure_ratio=ratio, temperature_ratio=temperature_ratio,
                inlet_temperature_k=temperature, isentropic_delta_t_k=delta_t,
                gamma=gamma, R=gas_constant, cp_j_kg_k=cp, cp_source="gamma_R_placeholder",
                specific_shaft_work_j_kg=work, design_mdot=mdot,
                fuel_mdot=mdot / (1 + mr), oxidizer_mdot=mdot * mr / (1 + mr))


def size_engine(cfg, combustion_properties):
    """Calculate the engine design and feed flows independently of tank construction."""
    prop, engine = cfg["prop_system"], cfg["engine"]
    Pc_target, MR_target = float(prop["Pc_target"]), float(prop["MR_target"])
    for name in ("Pc_target", "MR_target"):
        value = float(prop[name])
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    thrust_target = prop.get("thrust_target")
    if thrust_target is not None:
        thrust_target = float(thrust_target)
        if not isfinite(thrust_target) or thrust_target <= 0:
            raise ValueError("thrust_target must be finite and positive")
    elif "throat_area" not in engine:
        raise ValueError("Specify engine.throat_area or prop_system.thrust_target")
    cstar_efficiency = float(engine["cstar_efficiency"])
    cf_efficiency = float(engine["cf_efficiency"])
    if cstar_efficiency <= 0.0 or cf_efficiency <= 0.0:
        raise ValueError("Engine efficiencies must be positive")

    exit_pressure = engine.get("exit_pressure")
    if exit_pressure is not None and not 0.0 < float(exit_pressure) < Pc_target:
        raise ValueError("engine.exit_pressure must be between zero and Pc_target")

    if "expansion_ratio" in engine:
        expansion_ratio = float(engine["expansion_ratio"])
    else:
        if exit_pressure is None:
            raise ValueError("Specify engine.expansion_ratio or engine.exit_pressure")
        expansion_ratio = combustion_properties.expansion_ratio(
            Pc_target, MR_target, float(exit_pressure)
        )
    if not isfinite(expansion_ratio) or expansion_ratio < 1:
        raise ValueError("engine.expansion_ratio must be finite and >= 1")

    # With explicit epsilon there need not be a design exit-pressure input.
    # Sea level is then only the ambient reference for the design Cf/thrust.
    design_ambient_pressure = float(exit_pressure) if exit_pressure is not None else 101325.

    design = combustion_properties.evaluate(
        chamber_pressure=Pc_target,
        mixture_ratio=MR_target,
        ambient_pressure=design_ambient_pressure,
        expansion_ratio=expansion_ratio,
        cstar_efficiency=cstar_efficiency,
        cf_efficiency=cf_efficiency,
    )

    cstar = design.cstar
    Cf_design = design.Cf
    if cstar <= 0.0 or Cf_design <= 0.0:
        raise ValueError("Design cstar and thrust coefficient must be positive")

    combustion_gas = {
        "R": design.R,
        "gamma": design.gamma,
        "T": design.T,
    }

    throat_area = (float(engine["throat_area"]) if "throat_area" in engine
                        else thrust_target / (Pc_target * Cf_design))
    if not isfinite(throat_area) or throat_area <= 0:
        raise ValueError("engine.throat_area must be finite and positive")
    exit_area = throat_area * expansion_ratio
    design_thrust = Pc_target * throat_area * Cf_design
    mdot_total = Pc_target * throat_area / cstar
    mdot_ox = mdot_total * MR_target / (1.0 + MR_target)
    mdot_fuel = mdot_total / (1.0 + MR_target)
    flow_keys = ("design_mdot_oxidizer", "design_mdot_fuel")
    if any(key in prop for key in flow_keys):
        if not all(key in prop for key in flow_keys):
            raise ValueError("Specify both design_mdot_oxidizer and design_mdot_fuel")
        flows = [float(prop[key]) for key in flow_keys]
        if any(not isfinite(value) or value <= 0 for value in flows):
            raise ValueError("Design mass flows must be finite and positive")
        mdot_ox, mdot_fuel = flows
        mdot_total = sum(flows)
    nozzle_cd = float(prop["nozzle_cd"])
    return dict(
        Pc_target=Pc_target,
        MR_target=MR_target,
        thrust_target=thrust_target,
        cstar_efficiency=cstar_efficiency,
        cf_efficiency=cf_efficiency,
        expansion_ratio=expansion_ratio,
        design_ambient_pressure=design_ambient_pressure,
        cstar=cstar,
        Cf_design=Cf_design,
        combustion_gas=combustion_gas,
        throat_area=throat_area,
        exit_area=exit_area,
        design_thrust=design_thrust,
        mdot_total=mdot_total,
        mdot_ox=mdot_ox,
        mdot_fuel=mdot_fuel,
        nozzle_cd=nozzle_cd,
    )


def _feed_pressure_drop(mdot, cda, inlet_pressure, state, properties):
    """Invert liquid restriction sizing using the same upstream density convention."""
    def residual(dp):
        pressure = float(state.get("P", inlet_pressure + dp))
        rho = properties.state_pt(state["fluid"], pressure, float(state["T"])).rho
        return 2 * rho * dp * cda**2 - mdot**2

    if not isfinite(mdot) or mdot <= 0:
        raise ValueError("CdA feed sizing requires a positive finite design flow")
    maximum = (properties.state_bounds(state["fluid"])[0][1] - inlet_pressure
               if "P" not in state else float("inf"))
    if maximum <= 0:
        raise LookupBoundsError("Feed inlet pressure is outside the liquid property domain",
                                table=state["fluid"], axis="pressure", value=inlet_pressure)
    upper = min(maximum, max(1., inlet_pressure * .1))
    for _ in range(60):
        if residual(upper) >= 0:
            return float(brentq(residual, 0., upper, xtol=1e-7))
        if upper >= maximum:
            break
        upper = min(maximum, upper * 2)
    raise LookupBoundsError("Supplied feed CdA cannot deliver the design flow within the pressure domain",
                            table=state["fluid"], axis="pressure")


def prepare_design_inputs(cfg: dict, *, fluid_properties=None,
                          combustion_properties=None) -> tuple[dict, dict, dict]:
    """Resolve a finalized candidate once into template, P/T states and wall pressures.

    Propellant initial pressure can override design pressure. Pressurant initial
    pressure follows its design variable; inventories require tank geometry later.
    The returned dictionaries are independent of the source configuration.
    """
    cfg = deepcopy(cfg)
    prop = cfg["prop_system"]
    supplied = {}
    for leg in ("ox", "fuel"):
        dp_key, cda_key = f"{leg}_tank_inj_dp", f"{leg}_system_cda"
        dp, cda = prop.get(dp_key), prop.get(cda_key)
        if dp_key not in prop and cda_key not in prop:
            continue  # Other network templates need not have direct tank/injector feeds.
        if (dp is None) == (cda is None):
            raise ValueError(f"Specify exactly one of {dp_key} or {cda_key}")
        value = float(dp if dp is not None else cda)
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{dp_key if dp is not None else cda_key} must be finite and positive")
        if cda is not None:
            supplied[leg.upper() + "_TANK_INJ"] = value
            prop[dp_key] = 0.  # Bind the template; resolve this rise before sizing tanks.
    template = bind_template(cfg)
    nodes = template["nodes"]
    feeds = {}
    if supplied:
        if fluid_properties is None or combustion_properties is None:
            raise ValueError("CdA feed sizing requires fluid and combustion property sources")
        engine = size_engine(cfg, combustion_properties)
        for branch_id, cda in supplied.items():
            branch = template["branches"].get(branch_id)
            if branch is None or branch["component"] not in ("loss", "incompressible_loss"):
                raise ValueError(f"CdA feed sizing requires a direct liquid loss branch {branch_id}")
            node = nodes[branch["from"]]
            relation = node.get("P0")
            if (node["component"] != "propellant_tank" or branch.get("from_port") != "liquid"
                    or not isinstance(relation, dict) or relation.get("node") != branch["to"]
                    or set(relation) != {"node", "rise"} or relation["rise"] != 0.):
                raise ValueError(f"{branch_id} must feed an injector inlet with a tank pressure rise")
            if branch.get("CdA") is not None:
                raise ValueError(f"Specify CdA only in the vehicle config, not also in {branch_id}")
            circuit = template["circuits"][branch["circuit"]]
            flow = branch.get("design_mdot", circuit.get("design_mdot",
                engine["mdot_ox" if circuit["prop"] == "oxidizer" else "mdot_fuel"]))
            state = deepcopy(circuit.get("state0", prop["initial_conditions"][node["tank_id"]]))
            state["fluid"] = circuit.get("fluid", prop["initial_conditions"][node["tank_id"]]["fluid"])
            feeds[branch["from"]] = (float(flow), cda, state)
            branch["CdA"] = cda
    visiting = set()

    def pressure(key):
        if key in visiting:
            raise ValueError(f"Cyclic design-pressure references at {key!r}")
        if key not in nodes or "P0" not in nodes[key]:
            raise ValueError(f"Missing design pressure for node {key!r}")
        value = nodes[key]["P0"]
        if isinstance(value, dict):
            if "node" not in value or set(value) - {"node", "relative_rise", "rise", "drop"}:
                raise ValueError(f"Invalid design-pressure relation for {key!r}")
            visiting.add(key)
            if key in feeds:
                mdot, cda, state = feeds[key]
                value["rise"] = _feed_pressure_drop(
                    mdot, cda, pressure(value["node"]), state, fluid_properties)
            value = (pressure(value["node"]) * (1 + float(value.get("relative_rise", 0)))
                     + float(value.get("rise", 0)) - float(value.get("drop", 0)))
            visiting.remove(key)
        value = float(value)
        if not isfinite(value) or value <= 0:
            raise ValueError(f"Node {key!r} design pressure must be finite and positive")
        nodes[key]["P0"] = value
        return value

    for key, node in nodes.items():
        if "P0" in node:
            pressure(key)
        state = node.get('state0', {})
        if isinstance(state.get('P'), dict) and set(state['P']) == {'node'}:
            state['P'] = pressure(state['P']['node'])

    tank_nodes = {}
    for key, node in nodes.items():
        if "tank_id" in node:
            tank_id = node["tank_id"]
            if tank_id in tank_nodes or "P0" not in node:
                raise ValueError(f"Template must declare one design pressure for tank {tank_id!r}")
            tank_nodes[tank_id] = key
    bands = {}
    for branch in template["branches"].values():
        if branch["component"] == "bang_bang_valve":
            target, band = branch["to"], float(branch["pressure_band"])
            bands[target] = max(bands.get(target, band), band)
    wall_pressures = {}
    prop = cfg["prop_system"]
    states = deepcopy(prop.get("initial_conditions", {}))
    for tank_id, state in states.items():
        required = {"T", "gas_T"} if "gas_fluid" in state else {"T"}
        missing = required.difference(state)
        if missing:
            raise ValueError(f"Tank {tank_id!r} requires explicit temperatures: {sorted(missing)}")
        node_id = tank_nodes[tank_id]
        node = nodes[node_id]
        definition = cfg.get("tanks", {}).get(tank_id, {})
        pressurant = definition.get("type") == "pressurant" or node["component"] == "pressurant_tank"
        if pressurant:
            if "design_pressure" not in definition:
                raise ValueError(f"Pressurant tank {tank_id!r} requires tanks.{tank_id}.design_pressure")
            state["P"] = node["P0"]
        elif "P" not in state:
            state["P"] = node["P0"]
        for name in ("P", "T", "gas_T"):
            if name not in state:
                continue
            try:
                value = float(state[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Tank {tank_id!r} requires finite positive {name}") from exc
            if not isfinite(value) or value <= 0:
                raise ValueError(f"Tank {tank_id!r} requires finite positive {name}")
        nominal = max(float(node["P0"]), float(state["P"]))
        wall_pressures[tank_id] = nominal + bands[node_id] if node_id in bands else 1.1 * nominal
    return template, states, wall_pressures
