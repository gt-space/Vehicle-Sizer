from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple
from math import isfinite
from copy import deepcopy
from diagnostics.constraints import DesignInfeasible
from Fluids.helpers.battery import size_battery
from Fluids.helpers.pump_curve import scaled_pump_curve
from Fluids.design import initial_conditions, pump_definition, size_pump, size_gas_generator
from Fluids.helpers.templates import load_template
from Thermals.heat_sources import thermal_model

from .FluidNetwork import FluidNetwork
from Fluids import FluidsDef
from FluidTables.PropertyModels import (
    CEAPropertySource,
    CombustionPropertySource,
    CoolPropPropertySource,
    PureFluidProperties,
    PureFluidPropertySource,
    TableCombustionPropertySource,
)
from simulation_types import FluidOut, PropulsionOut


def _make_cea(engine_cfg: Dict[str, Any]):
    """Construct RocketCEA only when the CEA source is selected."""

    from rocketcea.cea_obj_w_units import CEA_Obj

    return CEA_Obj(
        oxName=engine_cfg["oxidizer"],
        fuelName=engine_cfg["fuel"],
        pressure_units="Pa",
        cstar_units="m/s",
        temperature_units="K",
        enthalpy_units="J/kg",
        density_units="kg/m^3",
        specific_heat_units="J/kg-K",
    )


class PropSystem:
    """Size and bind one declarative template, then expose the IDAS network to flight."""

    def __init__(self, cfg, tanks, fluid_properties=None, combustion_properties=None):
        self.design_config = deepcopy(cfg)
        if {'external_heating', 'nodes'} & self.design_config.get('thermal', {}).keys():
            raise ValueError('Select thermal.model per dynamic model instead of legacy global heating flags')
        self.cfg = self.design_config["prop_system"]
        self.template = load_template(self.design_config)
        self.cfg["template"] = self.template
        self.tank_definitions = self.design_config.get("tanks", {})
        self.initial_states = initial_conditions(self.design_config)
        self.engine_cfg = self.design_config["engine"]
        self.fluid_properties = fluid_properties if fluid_properties is not None else CoolPropPropertySource()
        self.combustion_properties = combustion_properties
        geometries = {key: tank.get_fluid_geometry() for key, tank in tanks.items()}
        self._initialize_tank_states(tanks, geometries)
        for name in ("Pc_target", "MR_target"):
            value = float(self.cfg[name])
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)
        self.thrust_target = self.cfg.get("thrust_target")
        if self.thrust_target is not None:
            self.thrust_target = float(self.thrust_target)
            if not isfinite(self.thrust_target) or self.thrust_target <= 0:
                raise ValueError("thrust_target must be finite and positive")
        elif "throat_area" not in self.engine_cfg:
            raise ValueError("Specify engine.throat_area or prop_system.thrust_target")
        self._size_engine()
        circuits, nodes, branches = self._bind_template(geometries)
        self.pump_sizing, self.gg_sizing, self.sizing_constraints = {}, {}, {}
        self._size_branches(circuits, nodes, branches)
        if set(self.pump_sizing) != set(self.cfg.get("pumps", {})):
            raise ValueError("Every configured pump must be referenced by exactly one template branch")
        if any(value < 0 for value in self.sizing_constraints.values()):
            raise DesignInfeasible(self.sizing_constraints, self.pump_sizing)
        self.battery_sizing = {}
        if "battery" in self.cfg:
            electric_pumps = [p for p in self.pump_sizing.values() if p["drive"] == "electric"]
            if not electric_pumps:
                raise ValueError("Battery sizing requires electric pumps")
            self.battery_sizing = size_battery(
                self.cfg["battery"],
                1000 * sum(pump["required_power_kw"] for pump in electric_pumps))
        self.circuits, self.node_definitions, self.branch_definitions = circuits, nodes, branches
        self._bind_outputs()
        self._configure_constraints()
        simulation = self.design_config.get("simulation", {})
        post_shutdown = simulation.get("fluid_solve_post_shutdown", True)
        triple = simulation.get("fluid_stop_at_triple_point", False)
        if not isinstance(post_shutdown, bool) or not isinstance(triple, bool):
            raise ValueError("Fluid stopping policies must be boolean")
        self.network = FluidNetwork(nodes, branches, fluid_properties=self.fluid_properties,
                                    combustion_properties=self.combustion_properties,
                                    tolerances=self.design_config.get("advanced", {}).get("fluid_network"),
                                    constraint_monitor=self._constraint_margins,
                                    stop_at_shutdown=not post_shutdown, stop_at_triple_point=triple)

    def _bind_template(self, geometries):
        template = deepcopy(self.template)
        circuits, nodes, branches = (template[k] for k in ("circuits", "nodes", "branches"))
        used = []
        for circuit in circuits.values():
            if "tank_id" in circuit:
                state = self.initial_states[circuit["tank_id"]]
                circuit.setdefault("fluid", state["fluid"])
                circuit.setdefault("state0", {"P": state["P"], "T": state["T"]})
            elif circuit["prop"] == "exhaust":
                circuit.setdefault("fluid", "combustion_gas")
                circuit.setdefault("state0", {"P": self.Pc_target, "T": self.combustion_gas["T"]})
        for key, node in nodes.items():
            kind = FluidNetwork._kind(node)
            node.pop("model", None)
            node["component"] = kind
            if kind in ("propellant_tank", "pressurant_tank"):
                tank_id = node["tank_id"]
                used.append(tank_id)
                state = deepcopy(self.initial_states[tank_id])
                node.update(geometry=geometries[tank_id], state0=state)
                if kind == 'propellant_tank':
                    node['pressure_tracking_target'] = float(node['P0'])
                if 'thermal' in self.tank_definitions.get(tank_id, {}):
                    if 'thermal' in node:
                        raise ValueError(f'Duplicate thermal configuration for tank {tank_id!r}')
                    node['thermal'] = deepcopy(self.tank_definitions[tank_id]['thermal'])
                if kind == "pressurant_tank":
                    node["fluid"] = state["fluid"]
                else:
                    node.update(liquid_fluid=state["fluid"], gas_fluid=state["gas_fluid"])
            elif kind == "combustor":
                for role in ("oxidizer", "fuel"):
                    fluids = {circuits[b["circuit"]]["fluid"] for b in branches.values()
                              if b["to"] == key and circuits[b["circuit"]]["prop"] == role}
                    if len(fluids) != 1:
                        raise ValueError(f"Combustor {key!r} needs one {role} fluid identity")
                    node.setdefault(f"{role}_fluid", fluids.pop())
                node.setdefault("combustion_fluid", "combustion_gas")
                node.update(expansion_ratio=self.expansion_ratio,
                            cstar_efficiency=self.cstar_efficiency, cf_efficiency=self.cf_efficiency)
        if len(used) != len(set(used)) or set(used) != set(geometries):
            raise ValueError("Template must reference every configured tank exactly once")
        for key, branch in branches.items():
            kind = FluidNetwork._kind(branch)
            branch.pop("model", None)
            branch["component"] = kind
            if kind == "nozzle":
                branch.setdefault("At", self.throat_area)
                branch.setdefault("Cd", self.nozzle_cd)
            elif kind != "switch_valve":
                branch.setdefault("CdA", None)
            for end in ("from", "to"):
                node = nodes[branch[end]]
                if node["component"] == "propellant_tank" and f"{end}_port" not in branch:
                    raise ValueError(f"Branch {key!r} must specify {end}_port for a propellant tank")
        return circuits, nodes, branches


    def _initialize_tank_states(
        self,
        tanks: Mapping[str, Any],
        geometries: Mapping[str, Any],
    ) -> None:
        """Convert configured P/T states into conserved mass and energy."""

        for tank_id, tank in tanks.items():
            try:
                state = self.initial_states[tank_id]
            except KeyError as error:
                raise ValueError(f"Tank {tank_id!r} requires an initial_conditions entry") from error
            geometry = geometries[tank_id]
            required = {"fluid", "P", "T"}
            if "gas_fluid" in state:
                required.add("gas_T")
            missing = required.difference(state)
            if missing:
                raise ValueError(
                    f"Tank {tank_id!r} initial_conditions is missing {sorted(missing)}"
                )
            pressure = float(state["P"])

            if "gas_fluid" not in state:
                conserved = ("m", "U")
                supplied = [name in state for name in conserved]
                if any(supplied) and not all(supplied):
                    raise ValueError(
                        f"Tank {tank_id!r} must supply both m and U or neither"
                    )
                fluid = self.fluid_properties.state_pt(
                    state["fluid"], pressure, float(state["T"])
                )
                state["m"] = fluid.rho * geometry.volume
                state["U"] = state["m"] * fluid.u
                continue

            conserved = ("m_liq", "U_liq", "m_ull", "U_ull")
            supplied = [name in state for name in conserved]
            if any(supplied) and not all(supplied):
                raise ValueError(
                    f"Tank {tank_id!r} must supply all conserved states or none"
                )
            if not hasattr(tank, "prop_mass"):
                raise ValueError(
                    f"Tank {tank_id!r} requires prop_mass to initialize from P/T"
                )

            liquid = self.fluid_properties.state_pt(
                state["fluid"], pressure, float(state["T"])
            )
            liquid_mass = float(tank.prop_mass)
            liquid_volume = liquid_mass / liquid.rho
            ullage_volume = geometry.volume - liquid_volume
            if ullage_volume <= 0.0:
                raise ValueError(f"Tank {tank_id!r} requires positive ullage volume")
            ullage = self.fluid_properties.state_pt(
                state["gas_fluid"], pressure, float(state["gas_T"])
            )
            state.update(
                {
                    "m_liq": liquid_mass,
                    "U_liq": liquid_mass * liquid.u,
                    "m_ull": ullage.rho * ullage_volume,
                    "U_ull": ullage.rho * ullage_volume * ullage.u,
                }
            )

    def _size_engine(self) -> None:
        """Calculate C*, Cf, throat area, and the design mixture mass flows."""

        self.cstar_efficiency = float(self.engine_cfg["cstar_efficiency"])
        self.cf_efficiency = float(self.engine_cfg["cf_efficiency"])
        if self.cstar_efficiency <= 0.0 or self.cf_efficiency <= 0.0:
            raise ValueError("Engine efficiencies must be positive")

        if self.combustion_properties is None:
            source = self.engine_cfg["property_source"]
            if source == "cea":
                self.cea = _make_cea(self.engine_cfg)
                self.combustion_properties = CEAPropertySource(self.cea)
            elif source == "table":
                self.combustion_properties = TableCombustionPropertySource(
                    lookup_file=self.engine_cfg["lookup_file"],
                    nfz=int(self.engine_cfg["nfz"]),
                )
            else:
                raise ValueError(f"Unknown engine.property_source={source!r}")

        exit_pressure = self.engine_cfg.get("exit_pressure")
        if exit_pressure is not None and not 0.0 < float(exit_pressure) < self.Pc_target:
            raise ValueError("engine.exit_pressure must be between zero and Pc_target")

        if "expansion_ratio" in self.engine_cfg:
            self.expansion_ratio = float(self.engine_cfg["expansion_ratio"])
        else:
            if exit_pressure is None:
                raise ValueError("Specify engine.expansion_ratio or engine.exit_pressure")
            self.expansion_ratio = self.combustion_properties.expansion_ratio(
                self.Pc_target, self.MR_target, float(exit_pressure)
            )
        if not isfinite(self.expansion_ratio) or self.expansion_ratio < 1:
            raise ValueError("engine.expansion_ratio must be finite and >= 1")

        # With explicit epsilon there need not be a design exit-pressure input.
        # Sea level is then only the ambient reference for the design Cf/thrust.
        self.design_ambient_pressure = float(exit_pressure) if exit_pressure is not None else 101325.

        design = self.combustion_properties.evaluate(
            chamber_pressure=self.Pc_target,
            mixture_ratio=self.MR_target,
            ambient_pressure=self.design_ambient_pressure,
            expansion_ratio=self.expansion_ratio,
            cstar_efficiency=self.cstar_efficiency,
            cf_efficiency=self.cf_efficiency,
        )

        self.cstar = design.cstar
        self.Cf_design = design.Cf
        if self.cstar <= 0.0 or self.Cf_design <= 0.0:
            raise ValueError("Design cstar and thrust coefficient must be positive")

        self.combustion_gas = {
            "R": design.R,
            "gamma": design.gamma,
            "T": design.T,
        }

        self.throat_area = (float(self.engine_cfg["throat_area"]) if "throat_area" in self.engine_cfg
                            else self.thrust_target / (self.Pc_target * self.Cf_design))
        if not isfinite(self.throat_area) or self.throat_area <= 0:
            raise ValueError("engine.throat_area must be finite and positive")
        self.exit_area = self.throat_area * self.expansion_ratio
        self.design_thrust = self.Pc_target * self.throat_area * self.Cf_design
        self.mdot_total = self.Pc_target * self.throat_area / self.cstar
        self.mdot_ox = self.mdot_total * self.MR_target / (1.0 + self.MR_target)
        self.mdot_fuel = self.mdot_total / (1.0 + self.MR_target)
        flow_keys = ("design_mdot_oxidizer", "design_mdot_fuel")
        if any(key in self.cfg for key in flow_keys):
            if not all(key in self.cfg for key in flow_keys):
                raise ValueError("Specify both design_mdot_oxidizer and design_mdot_fuel")
            flows = [float(self.cfg[key]) for key in flow_keys]
            if any(not isfinite(value) or value <= 0 for value in flows):
                raise ValueError("Design mass flows must be finite and positive")
            self.mdot_ox, self.mdot_fuel = flows
            self.mdot_total = sum(flows)
        self.nozzle_cd = float(self.cfg["nozzle_cd"])

    def _size_branches(
        self,
        circuits: Dict[str, Dict[str, Any]],
        nodes: Dict[str, Dict[str, Any]],
        branches: Dict[str, Dict[str, Any]],
    ) -> None:
        properties: Dict[str, PureFluidProperties] = {}

        def circuit_properties(circuit_id: str) -> PureFluidProperties:
            if circuit_id in properties:
                return properties[circuit_id]
            circuit = circuits[circuit_id]
            state0 = circuit["state0"]
            if "P" not in state0 or "T" not in state0:
                raise ValueError(
                    f"Circuit '{circuit_id}' requires a P/T design state"
                )
            properties[circuit_id] = self.fluid_properties.state_pt(
                circuit["fluid"],
                float(state0["P"]),
                float(state0["T"]),
            )
            return properties[circuit_id]

        design_flows = {
            "oxidizer": self.mdot_ox,
            "fuel": self.mdot_fuel,
            "exhaust": self.mdot_total,
        }
        def design_flow(branch):
            circuit = circuits[branch["circuit"]]
            flow = branch.get("design_mdot", circuit.get("design_mdot", design_flows.get(circuit["prop"])))
            if (flow is None or not isfinite(float(flow)) or float(flow) < 0
                    or (float(flow) == 0 and branch["component"] != "mass_flow")):
                raise ValueError(f"Branch {branch} requires a valid finite design_mdot")
            return float(flow)

        self._prepare_branch_design(circuits, branches)
        self._resolve_primary_design_flows(circuits, branches, design_flow, design_flows)

        self._size_pumps(circuits, nodes, branches, design_flow)
        self._size_gas_generators(circuits, nodes, branches)
        self._finalize_mass_flow_branches(branches)
        self._size_restrictions(circuits, nodes, branches, design_flow, circuit_properties)
        self._size_pressurant_feeds(circuits, nodes, branches, design_flow)

    @staticmethod
    def _prepare_branch_design(circuits, branches):
        for branch_id, branch in branches.items():
            if branch["circuit"] not in circuits:
                raise ValueError(f"Branch '{branch_id}' references unknown circuit '{branch['circuit']}'")
            branch["fluid"] = circuits[branch["circuit"]]["fluid"]
            if "require_choked" in branch:
                raise ValueError("require_choked was removed; configure tank pressure floors instead")

    @staticmethod
    def _resolve_primary_design_flows(circuits, branches, design_flow, design_flows):
        for branch in branches.values():
            if branch["component"] in ("mass_flow", "switch_valve", "relief_valve"):
                continue
            circuit = circuits[branch["circuit"]]
            if circuit["prop"] in design_flows or "design_mdot" in circuit:
                branch.setdefault("design_mdot", design_flow(branch))

    @staticmethod
    def _finalize_mass_flow_branches(branches):
        for branch in branches.values():
            if branch["component"] == "mass_flow":
                target = float(branch["target_mdot"])
                if not isfinite(target) or target < 0:
                    raise ValueError("Mass-flow target must be finite and nonnegative")
                if "design_mdot" in branch and float(branch["design_mdot"]) != target:
                    raise ValueError("Mass-flow design_mdot must equal target_mdot")
                branch["design_mdot"] = target

    def _size_pumps(self, circuits, nodes, branches, design_flow):
        for branch_id, branch in branches.items():
            branch_type = branch.get("component", branch.get("model"))
            circuit_id = branch["circuit"]
            circuit = circuits[circuit_id]
            if branch_type == "pump":
                pump_id = branch["pump_id"]
                if pump_id in self.pump_sizing:
                    raise ValueError(f"Pump {pump_id!r} must refer to one physical branch")
                if any(key in branch for key in ("dP", "head_model", "gas_CdA")) or branch.get("CdA") is not None:
                    raise ValueError("Pump head, curve and gas CdA belong in prop_system.pumps; branch overrides/internal losses are not supported")
                definition = pump_definition(self.cfg, pump_id)
                inlet_pressure = float(nodes[branch["from"]]["P0"])
                outlet_pressure = float(nodes[branch["to"]]["P0"])
                if abs(outlet_pressure - inlet_pressure - float(definition["pressure_rise_pa"])) > 1e-6 * max(outlet_pressure, 1):
                    raise ValueError(f"Pump {pump_id!r} design node pressures do not match its pressure rise")
                inlet_temperature = float(circuit["state0"]["T"])
                liquid = self.fluid_properties.state_pt(circuit["fluid"], inlet_pressure, inlet_temperature)
                sizing = size_pump(definition, design_flow(branch), liquid.rho)
                sizing.update(inlet_pressure_pa=inlet_pressure, inlet_temperature_k=inlet_temperature)
                self.pump_sizing[pump_id] = sizing
                if sizing["drive"] == "electric":
                    self.sizing_constraints[f"pump.{pump_id}.max_power"] = sizing["power_margin_kw"]
                branch.update(dP=sizing["pressure_rise_pa"], gas_CdA=float(definition["gas_CdA"]))
                if "curve" in definition:
                    curve = scaled_pump_curve(sizing["design_mdot"], sizing["pressure_rise_pa"], **definition["curve"])
                    branch["head_model"] = curve
                    sizing.update(curve_a=curve.a, curve_b=curve.b, curve_c=curve.c,
                                  curve_max_mdot=curve.max_mdot)
                continue


    def _size_restrictions(self, circuits, nodes, branches, design_flow, circuit_properties):
        for branch_id, branch in branches.items():
            branch_type = branch.get("component", branch.get("model"))
            circuit_id = branch["circuit"]
            circuit = circuits[circuit_id]
            if branch_type in ("loss", "incompressible_loss"):
                if branch["CdA"] is not None:
                    continue
                mdot = design_flow(branch)
                pressure_drop = (
                    nodes[branch["from"]]["P0"] - nodes[branch["to"]]["P0"]
                )
                branch["CdA"] = FluidsDef.incompressible_cda(
                    mdot,
                    circuit_properties(circuit_id).rho,
                    pressure_drop,
                )
                continue

            if branch_type == "compressible_loss":
                if branch["CdA"] is not None:
                    continue
                upstream_pressure = float(nodes[branch["from"]]["P0"])
                downstream_pressure = nodes[branch["to"]]["P0"]
                gas = circuit_properties(circuit_id)
                mdot = design_flow(branch)
                branch["CdA"] = FluidsDef.compressible_cda(
                    mdot,
                    upstream_pressure,
                    downstream_pressure,
                    gas.T,
                    gas.R,
                    gas.gamma,
                )
                continue


    def _size_pressurant_feeds(self, circuits, nodes, branches, design_flow):
        for branch_id, branch in branches.items():
            branch_type = branch.get("component", branch.get("model"))
            circuit_id = branch["circuit"]
            circuit = circuits[circuit_id]
            if branch_type in ("bang_bang_valve", "regulator"):
                if branch["CdA"] is not None:
                    continue
                target_node = nodes[branch["to"]]
                if target_node.get("component") != "propellant_tank":
                    raise ValueError(
                        f"Pressurant branch '{branch_id}' must feed a propellant tank"
                    )
                if target_node["gas_fluid"] != circuit["fluid"]:
                    raise ValueError(
                        f"Pressurant branch '{branch_id}' fluid must match tank ullage"
                    )
                liquid_branches = [
                    candidate
                    for candidate in branches.values()
                    if candidate["from"] == branch["to"]
                    and circuits[candidate["circuit"]]["prop"] in ("oxidizer", "fuel")
                    and candidate.get("from_port", "liquid") == "liquid"
                ]
                if not liquid_branches:
                    raise ValueError(
                        f"Tank '{branch['to']}' requires a liquid outlet"
                    )
                liquid_mdot = sum(design_flow(outlet) for outlet in liquid_branches)
                supply_branches = [b for b in branches.values() if b["to"] == branch["to"]
                                   and b.get("component", b.get("model")) in ("bang_bang_valve", "regulator")]
                shares = [float(b.get("demand_fraction", 1.0 if len(supply_branches) == 1 else float("nan"))) for b in supply_branches]
                if any(not isfinite(s) or s <= 0 for s in shares) or abs(sum(shares)-1) > 1e-9:
                    raise ValueError("Pressurant feeds to one tank require positive demand_fraction values summing to one")
                liquid_mdot *= float(branch.get("demand_fraction", 1.0))

                source_node = nodes[branch["from"]]
                if source_node.get("component") != "pressurant_tank":
                    raise ValueError(
                        f"Pressurant branch '{branch_id}' requires a gas-volume source"
                    )
                if source_node["fluid"] != circuit["fluid"]:
                    raise ValueError(
                        f"Pressurant branch '{branch_id}' fluid must match its source"
                    )
                downstream_pressure = float(target_node["P0"])
                source_state = source_node.get("state0", circuit["state0"])
                start_pressure = float(source_state["P"])
                initial_gas = self.fluid_properties.state_pt(source_node["fluid"], start_pressure, float(source_state["T"]))
                if downstream_pressure <= 0.0 or start_pressure <= 0.0:
                    raise ValueError(
                        f"Pressurant branch '{branch_id}' pressures must be positive"
                    )
                start_temperature = initial_gas.T
                gamma = initial_gas.gamma
                critical_ratio = (2.0 / (gamma + 1.0)) ** (
                    gamma / (gamma - 1.0)
                )
                eol_pressure = downstream_pressure / critical_ratio
                pressure_mid = 0.5 * (start_pressure + eol_pressure)
                tank_definition = getattr(self, "tank_definitions", {}).get(source_node.get("tank_id"), {})
                min_temperature = float(tank_definition.get("min_temperature", branch.get("min_temperature", float("nan"))))
                collapse_factor = float(branch["collapse_factor"])
                if not isfinite(min_temperature) or not isfinite(collapse_factor) or min_temperature <= 0.0 or collapse_factor <= 0.0:
                    raise ValueError(
                        f"Pressurant branch '{branch_id}' sizing inputs must be positive"
                    )
                temperature_mid = 0.5 * (start_temperature + min_temperature)
                gas = self.fluid_properties.state_pt(
                    circuit["fluid"],
                    pressure_mid,
                    temperature_mid,
                )
                gamma = gas.gamma
                gas_constant = gas.R
                if branch_type == "regulator":
                    capacity_factor = float(branch["capacity_factor"])
                    if not isfinite(capacity_factor) or capacity_factor < 1.0:
                        raise ValueError("Regulator capacity_factor must be finite and >= 1")
                    duty_cycle = 1.0 / capacity_factor
                else:
                    duty_cycle = float(branch["duty_cycle"])
                if not 0.0 < duty_cycle <= 1.0:
                    raise ValueError(
                        f"Gas branch '{branch_id}' duty cycle must be in (0, 1]"
                    )

                liquid_state = target_node.get("state0", circuits[liquid_branches[0]["circuit"]]["state0"])
                liquid_density = self.fluid_properties.state_pt(
                    target_node.get("liquid_fluid", circuits[liquid_branches[0]["circuit"]]["fluid"]),
                    float(liquid_state["P"]), float(liquid_state["T"]),
                ).rho
                liquid_vdot = liquid_mdot / liquid_density
                tank_gas = self.fluid_properties.state_pt(
                    circuit["fluid"],
                    downstream_pressure,
                    temperature_mid,
                )
                gas_mdot = (
                    collapse_factor * tank_gas.rho * liquid_vdot
                )

                branch["CdA"] = (
                    FluidsDef.compressible_cda(
                        gas_mdot,
                        pressure_mid,
                        downstream_pressure,
                        temperature_mid,
                        gas_constant,
                        gamma,
                    )
                    / duty_cycle
                )
                branch["design_mdot"] = gas_mdot / duty_cycle
                branch["design_pressure"] = pressure_mid
                branch["design_temperature"] = temperature_mid
                branch["eol_pressure"] = eol_pressure
                branch["target_pressure"] = downstream_pressure

    def _size_gas_generators(self, circuits, nodes, branches):
        assigned_pumps, assigned_branches = set(), set()
        for gg_id, definition in self.cfg.get("gas_generators", {}).items():
            pump_ids = definition["pumps"]
            if not pump_ids or len(set(pump_ids)) != len(pump_ids):
                raise ValueError("GG requires unique pump references")
            power = 0.0
            for pump_id in pump_ids:
                if pump_id in assigned_pumps or pump_id not in self.pump_sizing:
                    raise ValueError("GG pumps must exist and belong to exactly one drive")
                sizing = self.pump_sizing[pump_id]
                if sizing["drive"] != "gas_generator":
                    raise ValueError("GG requires gas_generator-driven pumps")
                assigned_pumps.add(pump_id)
                power += sizing["required_shaft_power_w"]
            feeds = {}
            for role in ("fuel", "oxidizer"):
                branch_id = definition[f"{role}_branch"]
                if branch_id in assigned_branches or branch_id not in branches:
                    raise ValueError("GG drains must exist and belong to exactly one drive")
                branch = branches[branch_id]
                if branch["component"] != "mass_flow" or circuits[branch["circuit"]]["prop"] != role:
                    raise ValueError("GG drain must be a mass_flow branch of the matching propellant")
                source = nodes[branch["from"]]
                if source["component"] != "propellant_tank" or branch.get("from_port") != "liquid":
                    raise ValueError("GG approximation currently requires direct tank liquid drains")
                if any(k in branch for k in ("target_mdot", "design_mdot")):
                    raise ValueError("GG drain flows are calculated; remove explicit flow overrides")
                assigned_branches.add(branch_id)
                feeds[role] = branch
            stiffness = float(definition["stiffness"])
            if not isfinite(stiffness) or stiffness < 0:
                raise ValueError("GG stiffness must be finite and nonnegative")
            pressures = {role: float(nodes[b["from"]]["P0"]) for role, b in feeds.items()}
            pressure = min(pressures.values()) / (1 + stiffness)
            # Reuse design nozzle coordinates; GG sizing only consumes chamber T, gamma and R.
            properties = self.combustion_properties.evaluate(
                chamber_pressure=pressure,
                mixture_ratio=float(definition["mixture_ratio"]),
                ambient_pressure=self.design_ambient_pressure,
                expansion_ratio=self.expansion_ratio,
            ).as_dict()
            sizing = size_gas_generator(power, pressure, definition, properties)
            sizing.update(upstream_pressures_pa=pressures,
                          feed_stiffness={role: p / pressure - 1 for role, p in pressures.items()})
            self.gg_sizing[gg_id] = sizing
            for role, branch in feeds.items():
                branch.update(target_mdot=sizing[f"{role}_mdot"], design_mdot=sizing[f"{role}_mdot"],
                              required_phase="liquid")
        turbine_pumps = {k for k, p in self.pump_sizing.items() if p["drive"] == "gas_generator"}
        if assigned_pumps != turbine_pumps:
            raise ValueError("Every gas_generator pump must belong to one GG drive")

    def _bind_outputs(self):
        """Discover engine reporting connections; IDs belong to templates."""
        chambers = [key for key, node in self.node_definitions.items() if node.get("component") == "combustor"]
        if len(chambers) != 1:
            raise ValueError("The current engine model requires one combustor per vehicle")
        self.chamber_id = chambers[0]
        self.ambient_id = self.node_definitions[self.chamber_id]["ambient_node"]
        self.engine_feeds = {role: [] for role in ("oxidizer", "fuel")}
        nozzles = []
        for key, branch in self.branch_definitions.items():
            role = self.circuits[branch["circuit"]]["prop"]
            if branch["to"] == self.chamber_id and role in self.engine_feeds:
                self.engine_feeds[role].append(key)
            if branch["from"] == self.chamber_id and branch.get("component") == "nozzle":
                nozzles.append(key)
        if len(nozzles) != 1 or not all(self.engine_feeds.values()):
            raise ValueError("Combustor requires fuel/oxidizer feeds and one nozzle")
        self.nozzle_id = nozzles[0]

    def _configure_constraints(self):
        """Attach owner-defined temperature and absolute pressure floors to tanks."""
        self.tank_limits = {}
        initial = dict(self.sizing_constraints)
        for node_id, node in self.node_definitions.items():
            tank_id = node.get("tank_id")
            if tank_id is None:
                continue
            definition = self.tank_definitions.get(tank_id, {})
            limits = {name: float(definition[name]) for name in ("min_temperature", "min_pressure") if name in definition}
            if node.get("component") == "pressurant_tank" and "min_temperature" not in limits:
                branch_limits = [float(b["min_temperature"]) for b in self.branch_definitions.values()
                          if b["from"] == node_id and "min_temperature" in b]
                if branch_limits:
                    limits["min_temperature"] = max(branch_limits)
                else:
                    raise ValueError(f"Pressurant tank {tank_id!r} requires min_temperature")
            if any(not isfinite(v) or v <= 0 for v in limits.values()):
                raise ValueError(f"Tank {tank_id!r} limits must be finite and positive")
            self.tank_limits[node_id] = (tank_id, limits)
            state = node["state0"]
            if "min_temperature" in limits:
                initial[f"tank.{tank_id}.Tmin"] = min(float(state[key]) for key in ("T", "gas_T") if key in state) - limits["min_temperature"]
            if "min_pressure" in limits:
                initial[f"tank.{tank_id}.Pmin"] = float(state["P"]) - limits["min_pressure"]
        self.initial_constraints = initial
        if any(value < 0 for value in initial.values()):
            raise DesignInfeasible(initial, self.pump_sizing)

    def _constraint_margins(self, state):
        """Evaluate owner-defined hard floors on accepted fluid states."""
        for branch_id, definition in self.branch_definitions.items():
            curve = definition.get("head_model")
            if curve is None:
                continue
            branch_state = state.branches[branch_id]
            if branch_state.get("pump_active", False):
                curve.validate_flow(float(branch_state["mdot"]))
        margins = {}
        for node_id, (tank_id, limits) in self.tank_limits.items():
            node = state.nodes[node_id]
            if "min_temperature" in limits:
                temperatures = [float(fluid["T"]) for fluid in node.get("fluids", {}).values() if "T" in fluid]
                if "T" in node:
                    temperatures.append(float(node["T"]))
                if not temperatures:
                    raise ValueError(f"Tank {tank_id!r} has no temperature to monitor")
                margins[f"tank.{tank_id}.Tmin"] = min(temperatures) - limits["min_temperature"]
            if "min_pressure" in limits:
                margins[f"tank.{tank_id}.Pmin"] = float(node["P"]) - limits["min_pressure"]
        if any(not isfinite(v) for v in margins.values()):
            raise ValueError("Propulsion constraint margins must be finite")
        return margins

    def _propulsion_output(
        self,
        network_output: Dict[str, Any],
    ) -> PropulsionOut:
        mdot = network_output["mdot"]
        mdot_ox = sum(mdot[key] for key in self.engine_feeds["oxidizer"])
        mdot_fuel = sum(mdot[key] for key in self.engine_feeds["fuel"])
        chamber = network_output["node"][self.chamber_id]
        Pc = chamber["P"]
        MR = chamber["MR"]
        Cf = chamber["Cf"]
        mode = chamber["mode"]
        nozzle = network_output["branch"][self.nozzle_id]
        return PropulsionOut(
            mode=mode,
            shutdown_reason=chamber["shutdown_reason"],
            thrust=(
                Pc * self.throat_area * Cf
                if mode == "combusting"
                else nozzle["thrust"]
            ),
            Pc=Pc,
            MR=MR,
            Cf=Cf,
            cstar=chamber["cstar"],
            mdot_ox=mdot_ox,
            mdot_fuel=mdot_fuel,
            mdot_nozzle=network_output["mdot"][self.nozzle_id],
        )

    def update(
        self,
        dt: Optional[float],
        atm: Any,
        heat_rate: Dict[str, Any],
        bcs: Optional[Dict[str, Dict[str, Any]]] = None,
        commit: bool = True,
        axial_specific_force: float = 0.0,
    ) -> FluidOut:
        boundaries = dict(bcs or {})
        boundaries[self.ambient_id] = {"P": float(atm.p)}

        event_start = len(self.network.events)
        heat_rate = dict(heat_rate or {})
        aero_nodes = {key for key, node in self.node_definitions.items()
                      if thermal_model(node.get('thermal')) == 'Aeroheating'}
        if set(heat_rate) - aero_nodes:
            raise ValueError('External heat rates require Aeroheating on the receiving node')
        for key in aero_nodes - heat_rate.keys():
            if dt is None or dt == 0:
                heat_rate[key] = {}  # Initial fluid snapshot precedes the first wall trial.
            else:
                raise ValueError(f'Aeroheating node {key!r} requires wall heat rates')
        result = self.network.update(
            dt=dt,
            bcs=boundaries,
            heat_rate=heat_rate,
            commit=commit,
            axial_specific_force=axial_specific_force,
        )
        margins, times = dict(result["constraints"]), dict(result["constraint_times"])
        for key, value in self.initial_constraints.items():
            if value < margins.get(key, float("inf")):
                margins[key], times[key] = value, 0.0
        counts = {}
        events = []
        # Count the full returned history so previews and restored steps agree.
        for event in result["events"][:event_start]:
            key = f"{event['kind']}:{event['component']}:{event['name']}"
            counts[key] = counts.get(key, 0) + 1
        for event in result["events"][event_start:]:
            key = f"{event['kind']}:{event['component']}:{event['name']}"
            counts[key] = counts.get(key, 0) + 1
            event = {**event, "event": event["name"], "count": counts[key]}
            if "before" in event:
                before = event["before"]
                event["before"] = FluidOut(
                    node=before["node"], branch=before["branch"], mdot=before["mdot"],
                    td_state={k: before["node"][k] for k in result["td_state"]},
                    propulsion=self._propulsion_output(before))
            events.append(event)
        return FluidOut(
            node=result["node"],
            branch=result["branch"],
            td_state=result["td_state"],
            mdot=result["mdot"],
            propulsion=self._propulsion_output(result),
            events=tuple(events),
            event_counts=result["event_counts"],
            constraints=margins,
            constraint_times=times,
        )

    def checkpoint(self):
        return self.network._checkpoint()

    def restore(self, checkpoint):
        self.network.restore(checkpoint)

    def close(self):
        self.network.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
