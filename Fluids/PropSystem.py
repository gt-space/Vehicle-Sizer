from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple
from math import isfinite
from copy import deepcopy
from constraints import DesignInfeasible
from .pump_curve import scaled_pump_curve
from .design import initial_conditions, pressure_ladder, tank_design_pressure, pump_definition, size_electric_pump

from .FluidNetwork import FluidNetwork
from . import FluidsDef
from FluidProperties.PropertyModels import (
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
    """Configure, size, and expose the vehicle fluid network."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        tanks: Mapping[str, Any],
        fluid_properties: Optional[PureFluidPropertySource] = None,
        combustion_properties: Optional[CombustionPropertySource] = None,
    ):
        self.cfg = deepcopy(cfg["prop_system"])
        self.tank_definitions = deepcopy(cfg.get("tanks", {}))
        self.design_config = cfg
        self.initial_states = initial_conditions(cfg)
        self.engine_cfg = cfg["engine"]
        legacy = self.cfg.get("press_model")
        self.feed_type = self.cfg.get(
            "feed_type",
            "pressure_fed" if legacy == "blowdown" else legacy,
        )
        self.pressurization = self.cfg.get(
            "pressurization",
            "blowdown" if legacy == "blowdown" else "bang_bang",
        )
        if self.feed_type not in ("pressure_fed", "pump_fed"):
            raise ValueError(f"Unknown feed_type={self.feed_type!r}")
        if self.pressurization not in ("bang_bang", "regulator", "blowdown"):
            raise ValueError(f"Unknown pressurization={self.pressurization!r}")
        self.fluid_properties = (
            fluid_properties
            if fluid_properties is not None
            else CoolPropPropertySource()
        )
        self.combustion_properties = combustion_properties
        tank_geometries = {
            tank_id: tank.get_fluid_geometry() for tank_id, tank in tanks.items()
        }
        self._initialize_tank_states(tanks, tank_geometries)

        self.Pc_target = float(self.cfg["Pc_target"])
        self.MR_target = float(self.cfg["MR_target"])
        self.thrust_target = float(self.cfg["thrust_target"])
        self.fuel_inj_stiffness = float(self.cfg["fuel_inj_stiffness"])
        self.ox_inj_stiffness = float(self.cfg["ox_inj_stiffness"])

        positive = {
            "Pc_target": self.Pc_target,
            "MR_target": self.MR_target,
            "thrust_target": self.thrust_target,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"PropSystem inputs must be positive: {invalid}")
        if min(self.fuel_inj_stiffness, self.ox_inj_stiffness) < 0.0:
            raise ValueError("Injector stiffness values must be nonnegative")

        if self.feed_type == "pump_fed":
            self.fuel_inj_pumpout_dp = float(self.cfg["fuel_inj_pumpout_dp"])
            self.ox_inj_pumpout_dp = float(self.cfg["ox_inj_pumpout_dp"])
            self.fuel_pumpin_tank_dp = float(self.cfg["fuel_pumpin_tank_dp"])
            self.ox_pumpin_tank_dp = float(self.cfg["ox_pumpin_tank_dp"])
        else:
            self.fuel_tank_inj_dp = float(self.cfg["fuel_tank_inj_dp"])
            self.ox_tank_inj_dp = float(self.cfg["ox_tank_inj_dp"])

        self._size_engine()
        self.target_ladder = self._build_pressure_ladder()

        circuits, nodes, branches = self._wire_network(tank_geometries)
        self.pump_sizing = {}
        self.sizing_constraints = {}
        self._size_branches(circuits, nodes, branches)
        if set(self.pump_sizing) != set(self.cfg.get("pumps", {})):
            raise ValueError("Every configured pump must be referenced by exactly one template branch")
        if any(value < 0 for value in self.sizing_constraints.values()):
            raise DesignInfeasible(self.sizing_constraints, self.pump_sizing)
        self.node_definitions = nodes
        self.branch_definitions = branches
        self.circuits = circuits
        self._bind_outputs()
        self._configure_constraints()
        post_shutdown = cfg.get("simulation", {}).get("fluid_solve_post_shutdown", True)
        if not isinstance(post_shutdown, bool):
            raise ValueError("simulation.fluid_solve_post_shutdown must be boolean")
        self.network = FluidNetwork(
            nodes=nodes,
            branches=branches,
            fluid_properties=self.fluid_properties,
            combustion_properties=self.combustion_properties,
            tolerances=cfg.get("advanced", {}).get("fluid_network"),
            constraint_monitor=self._constraint_margins,
            stop_at_shutdown=not post_shutdown,
            stop_at_triple_point=cfg.get("simulation", {}).get("fluid_stop_at_triple_point", False),
        )

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
                raise ValueError(f"Tank {tank_id!r} requires a state0 entry") from error
            geometry = geometries[tank_id]
            required = {"fluid", "P", "T"}
            if "gas_fluid" in state:
                required.add("gas_T")
            missing = required.difference(state)
            if missing:
                raise ValueError(
                    f"Tank {tank_id!r} state0 is missing {sorted(missing)}"
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
            if all(supplied) and not hasattr(tank, "prop_mass"):
                # Compatibility for geometry-only network harnesses. Vehicle
                # builds always derive inventories from propellant_mass and P/T.
                continue
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

        exit_pressure = float(self.engine_cfg["exit_pressure"])
        if not 0.0 < exit_pressure < self.Pc_target:
            raise ValueError("engine.exit_pressure must be between zero and Pc_target")

        self.expansion_ratio = self.combustion_properties.expansion_ratio(
            self.Pc_target, self.MR_target, exit_pressure
        )

        design = self.combustion_properties.evaluate(
            chamber_pressure=self.Pc_target,
            mixture_ratio=self.MR_target,
            ambient_pressure=exit_pressure,
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

        self.throat_area = self.thrust_target / (self.Pc_target * self.Cf_design)
        self.exit_area = self.throat_area * self.expansion_ratio
        self.mdot_total = self.Pc_target * self.throat_area / self.cstar
        self.mdot_ox = self.mdot_total * self.MR_target / (1.0 + self.MR_target)
        self.mdot_fuel = self.mdot_total / (1.0 + self.MR_target)
        self.nozzle_cd = float(self.cfg["nozzle_cd"])

    def _build_pressure_ladder(self) -> Dict[str, float]:
        return pressure_ladder(self.cfg)

    def _state0(self, name: str) -> Dict[str, Any]:
        return dict(self.initial_states[name])

    def _wire_network(
        self,
        tank_geometries: Mapping[str, Any],
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]],
    ]:
        if "template" in self.cfg:
            template = self._configured_template(tank_geometries)
        elif self.feed_type == "pump_fed":
            template = self._template_pump_fed(tank_geometries)
        elif self.pressurization != "blowdown":
            template = self._template_pressure_fed(tank_geometries)
        else:
            template = self._template_blowdown(tank_geometries)
        circuits, nodes, branches = template
        for branch in branches.values():
            source = nodes[branch["from"]].get("component")
            target = nodes[branch["to"]].get("component")
            circuit = circuits[branch["circuit"]]["prop"]
            if source == "pressurant_tank":
                branch.setdefault("from_port", "gas")
            elif source == "propellant_tank":
                branch.setdefault("from_port", "liquid" if circuit in ("oxidizer", "fuel") else "ullage")
            elif source == "combustor":
                branch.setdefault("from_port", "nozzle")
            if target == "propellant_tank":
                branch.setdefault("to_port", "ullage")
            elif target == "combustor" and circuit in ("oxidizer", "fuel"):
                branch.setdefault("to_port", circuit)
        return circuits, nodes, branches

    def _configured_template(self, tank_geometries):
        """Bind a declarative wiring template to sized tank/engine data."""
        template = deepcopy(self.cfg["template"])
        circuits, nodes, branches = (template[key] for key in ("circuits", "nodes", "branches"))
        used = []
        for node in nodes.values():
            kind = FluidNetwork._kind(node)
            node.pop("model", None)
            node["component"] = kind
            if kind in ("propellant_tank", "pressurant_tank"):
                tank_id = node["tank_id"]
                used.append(tank_id)
                state = self._state0(tank_id)
                node.update(geometry=tank_geometries[tank_id], state0=state)
                if kind == "pressurant_tank":
                    node["P0"] = state["P"]
                if "P0" not in node:
                    node["P0"] = tank_design_pressure(self.design_config, tank_id, fallback=state["P"])
                if kind == "pressurant_tank":
                    node["fluid"] = state["fluid"]
                else:
                    node.update(liquid_fluid=state["fluid"], gas_fluid=state["gas_fluid"])
            elif kind == "combustor":
                node.setdefault("P0", self.Pc_target)
                node.update(expansion_ratio=self.expansion_ratio,
                            cstar_efficiency=self.cstar_efficiency, cf_efficiency=self.cf_efficiency)
            if isinstance(node.get("P0"), str):
                node["P0"] = self.target_ladder[node["P0"]]
        if len(used) != len(set(used)) or set(used) != set(tank_geometries):
            raise ValueError("Template must reference every configured tank exactly once")
        for circuit in circuits.values():
            if "tank_id" in circuit:
                state = self._state0(circuit["tank_id"])
                circuit.setdefault("fluid", state["fluid"])
                circuit.setdefault("state0", {"P": state["P"], "T": state["T"]})
            elif circuit["prop"] == "exhaust":
                circuit.setdefault("fluid", "combustion_gas")
                circuit.setdefault("state0", {"P": self.Pc_target, "T": self.combustion_gas["T"]})
        for branch in branches.values():
            kind = FluidNetwork._kind(branch)
            branch.pop("model", None)
            branch["component"] = kind
            if branch.get("component") == "nozzle":
                branch.setdefault("At", self.throat_area)
                branch.setdefault("Cd", self.nozzle_cd)
            else:
                branch.setdefault("CdA", None)
        return circuits, nodes, branches

    def _pressurant_branches(self):
        """Both active pressurization options share routing and capacity sizing."""
        if self.pressurization == "blowdown":
            return {}
        regulator = self.pressurization == "regulator"
        suffix = "REGULATOR" if regulator else "BANGBANG"
        return {
            f"{leg}_{suffix}": {
                "component": "regulator" if regulator else "bang_bang_valve",
                "circuit": "pressurant", "from": "press_tank", "to": destination,
                **self.cfg[self.pressurization][f"{leg}_{suffix}"], "CdA": None,
            }
            for leg, destination in (("OX", "ox_ullage"), ("FUEL", "fuel_ullage"))
        }

    def _template_pressure_fed(self, tank_geometries: Mapping[str, Any]):
        if "press_tank" not in tank_geometries:
            raise ValueError("Pressure-fed template requires tank 'press_tank'")
        ox_state0 = self._state0("ox_tank")
        fuel_state0 = self._state0("fuel_tank")
        press_state0 = self._state0("press_tank")
        ox_fluid = str(ox_state0["fluid"])
        fuel_fluid = str(fuel_state0["fluid"])
        press_fluid = str(press_state0["fluid"])
        circuits = {
            "oxidizer": {
                "prop": "oxidizer",
                "fluid": ox_fluid,
                "state0": {"P": ox_state0["P"], "T": ox_state0["T"]},
            },
            "fuel": {
                "prop": "fuel",
                "fluid": fuel_fluid,
                "state0": {"P": fuel_state0["P"], "T": fuel_state0["T"]},
            },
            "pressurant": {
                "prop": "pressurant",
                "fluid": press_fluid,
                "state0": {"P": press_state0["P"], "T": press_state0["T"]},
            },
            "combustion_gas": {
                "prop": "exhaust",
                "fluid": "combustion_gas",
                "state0": {"P": self.Pc_target, "T": self.combustion_gas["T"]},
            },
        }
        nodes = {
            "press_tank": {
                "component": "pressurant_tank",
                "tank_id": "press_tank",
                "fluid": press_fluid,
                "geometry": tank_geometries["press_tank"],
                "P0": float(press_state0["P"]),
                "state0": press_state0,
            },
            "ox_ullage": {
                "component": "propellant_tank",
                "tank_id": "ox_tank",
                "liquid_fluid": ox_fluid,
                "gas_fluid": press_fluid,
                "geometry": tank_geometries["ox_tank"],
                "P0": self.target_ladder["Pox_tank"],
                "state0": ox_state0,
            },
            "ox_inj_in": {
                "model": "junction",
                "P0": self.target_ladder["Pox_inj"],
            },
            "fuel_ullage": {
                "component": "propellant_tank",
                "tank_id": "fuel_tank",
                "liquid_fluid": fuel_fluid,
                "gas_fluid": press_fluid,
                "geometry": tank_geometries["fuel_tank"],
                "P0": self.target_ladder["Pfuel_tank"],
                "state0": fuel_state0,
            },
            "fuel_inj_in": {
                "model": "junction",
                "P0": self.target_ladder["Pfuel_inj"],
            },
            "thrust_chamber": {
                "component": "combustor",
                "P0": self.Pc_target,
                "oxidizer_fluid": ox_fluid,
                "fuel_fluid": fuel_fluid,
                "combustion_fluid": "combustion_gas",
                "ambient_node": "ambient",
                "expansion_ratio": self.expansion_ratio,
                "cstar_efficiency": self.cstar_efficiency,
                "cf_efficiency": self.cf_efficiency,
            },
            "ambient": {"model": "boundary"},
        }
        branches = {
            **self._pressurant_branches(),
            "OX_TANK_INJ": {
                "component": "loss",
                "circuit": "oxidizer",
                "from": "ox_ullage",
                "to": "ox_inj_in",
                "CdA": None,
            },
            "OX_INJ": {
                "component": "loss",
                "circuit": "oxidizer",
                "from": "ox_inj_in",
                "to": "thrust_chamber",
                "CdA": None,
            },
            "FUEL_TANK_INJ": {
                "component": "loss",
                "circuit": "fuel",
                "from": "fuel_ullage",
                "to": "fuel_inj_in",
                "CdA": None,
            },
            "FUEL_INJ": {
                "component": "loss",
                "circuit": "fuel",
                "from": "fuel_inj_in",
                "to": "thrust_chamber",
                "CdA": None,
            },
            "NOZZLE": {
                "component": "nozzle",
                "circuit": "combustion_gas",
                "from": "thrust_chamber",
                "to": "ambient",
                "At": self.throat_area,
                "Cd": self.nozzle_cd,
            },
        }
        return circuits, nodes, branches

    def _template_blowdown(self, tank_geometries: Mapping[str, Any]):
        ox_state0 = self._state0("ox_tank")
        fuel_state0 = self._state0("fuel_tank")
        ox_fluid = str(ox_state0["fluid"])
        fuel_fluid = str(fuel_state0["fluid"])
        circuits = {
            "oxidizer": {
                "prop": "oxidizer",
                "fluid": ox_fluid,
                "state0": {"P": ox_state0["P"], "T": ox_state0["T"]},
            },
            "fuel": {
                "prop": "fuel",
                "fluid": fuel_fluid,
                "state0": {"P": fuel_state0["P"], "T": fuel_state0["T"]},
            },
            "combustion_gas": {
                "prop": "exhaust",
                "fluid": "combustion_gas",
                "state0": {"P": self.Pc_target, "T": self.combustion_gas["T"]},
            },
        }
        nodes = {
            "ox_ullage": {
                "component": "propellant_tank",
                "tank_id": "ox_tank",
                "liquid_fluid": ox_fluid,
                "gas_fluid": str(ox_state0["gas_fluid"]),
                "geometry": tank_geometries["ox_tank"],
                "P0": self.target_ladder["Pox_tank"],
                "state0": ox_state0,
            },
            "ox_inj_in": {
                "model": "junction",
                "P0": self.target_ladder["Pox_inj"],
            },
            "fuel_ullage": {
                "component": "propellant_tank",
                "tank_id": "fuel_tank",
                "liquid_fluid": fuel_fluid,
                "gas_fluid": str(fuel_state0["gas_fluid"]),
                "geometry": tank_geometries["fuel_tank"],
                "P0": self.target_ladder["Pfuel_tank"],
                "state0": fuel_state0,
            },
            "fuel_inj_in": {
                "model": "junction",
                "P0": self.target_ladder["Pfuel_inj"],
            },
            "thrust_chamber": {
                "component": "combustor",
                "P0": self.Pc_target,
                "oxidizer_fluid": ox_fluid,
                "fuel_fluid": fuel_fluid,
                "combustion_fluid": "combustion_gas",
                "ambient_node": "ambient",
                "expansion_ratio": self.expansion_ratio,
                "cstar_efficiency": self.cstar_efficiency,
                "cf_efficiency": self.cf_efficiency,
            },
            "ambient": {"model": "boundary"},
        }
        branches = {
            "OX_TANK_INJ": {
                "component": "loss",
                "circuit": "oxidizer",
                "from": "ox_ullage",
                "to": "ox_inj_in",
                "CdA": None,
            },
            "OX_INJ": {
                "component": "loss",
                "circuit": "oxidizer",
                "from": "ox_inj_in",
                "to": "thrust_chamber",
                "CdA": None,
            },
            "FUEL_TANK_INJ": {
                "component": "loss",
                "circuit": "fuel",
                "from": "fuel_ullage",
                "to": "fuel_inj_in",
                "CdA": None,
            },
            "FUEL_INJ": {
                "component": "loss",
                "circuit": "fuel",
                "from": "fuel_inj_in",
                "to": "thrust_chamber",
                "CdA": None,
            },
            "NOZZLE": {
                "component": "nozzle",
                "circuit": "combustion_gas",
                "from": "thrust_chamber",
                "to": "ambient",
                "At": self.throat_area,
                "Cd": self.nozzle_cd,
            },
        }
        return circuits, nodes, branches

    def _template_pump_fed(self, tank_geometries: Mapping[str, Any]):
        actively_pressurized = self.pressurization != "blowdown"
        if actively_pressurized and "press_tank" not in tank_geometries:
            raise ValueError("Active pressurization requires tank 'press_tank'")
        ox_state0 = self._state0("ox_tank")
        fuel_state0 = self._state0("fuel_tank")
        press_state0 = self._state0("press_tank") if actively_pressurized else None
        ox_fluid = str(ox_state0["fluid"])
        fuel_fluid = str(fuel_state0["fluid"])
        press_fluid = str(press_state0["fluid"]) if press_state0 else None
        circuits = {
            "oxidizer": {
                "prop": "oxidizer",
                "fluid": ox_fluid,
                "state0": {"P": ox_state0["P"], "T": ox_state0["T"]},
            },
            "fuel": {
                "prop": "fuel",
                "fluid": fuel_fluid,
                "state0": {"P": fuel_state0["P"], "T": fuel_state0["T"]},
            },
            **({
                "pressurant": {
                    "prop": "pressurant",
                    "fluid": press_fluid,
                    "state0": {"P": press_state0["P"], "T": press_state0["T"]},
                },
            } if actively_pressurized else {}),
            "combustion_gas": {
                "prop": "exhaust",
                "fluid": "combustion_gas",
                "state0": {"P": self.Pc_target, "T": self.combustion_gas["T"]},
            },
        }
        nodes = {
            **({
                "press_tank": {
                    "component": "pressurant_tank",
                    "tank_id": "press_tank",
                    "fluid": press_fluid,
                    "geometry": tank_geometries["press_tank"],
                    "P0": float(press_state0["P"]),
                    "state0": press_state0,
                },
            } if actively_pressurized else {}),
            "ox_ullage": {
                "component": "propellant_tank",
                "tank_id": "ox_tank",
                "liquid_fluid": ox_fluid,
                "gas_fluid": press_fluid or str(ox_state0["gas_fluid"]),
                "geometry": tank_geometries["ox_tank"],
                "P0": self.target_ladder["Pox_tank"],
                "state0": ox_state0,
            },
            "ox_pump_in": {
                "model": "junction",
                "P0": self.target_ladder["Pox_pump_inlet"],
            },
            "ox_pump_out": {
                "model": "junction",
                "P0": self.target_ladder["Pox_pump_outlet"],
            },
            **({
                "ox_inj_in": {
                    "model": "junction",
                    "P0": self.target_ladder["Pox_inj"],
                },
            } if self.ox_inj_stiffness > 0.0 else {}),
            "fuel_ullage": {
                "component": "propellant_tank",
                "tank_id": "fuel_tank",
                "liquid_fluid": fuel_fluid,
                "gas_fluid": press_fluid or str(fuel_state0["gas_fluid"]),
                "geometry": tank_geometries["fuel_tank"],
                "P0": self.target_ladder["Pfuel_tank"],
                "state0": fuel_state0,
            },
            "fuel_pump_in": {
                "model": "junction",
                "P0": self.target_ladder["Pfuel_pump_inlet"],
            },
            "fuel_pump_out": {
                "model": "junction",
                "P0": self.target_ladder["Pfuel_pump_outlet"],
            },
            **({
                "fuel_inj_in": {
                    "model": "junction",
                    "P0": self.target_ladder["Pfuel_inj"],
                },
            } if self.fuel_inj_stiffness > 0.0 else {}),
            "thrust_chamber": {
                "component": "combustor",
                "P0": self.Pc_target,
                "oxidizer_fluid": ox_fluid,
                "fuel_fluid": fuel_fluid,
                "combustion_fluid": "combustion_gas",
                "ambient_node": "ambient",
                "expansion_ratio": self.expansion_ratio,
                "cstar_efficiency": self.cstar_efficiency,
                "cf_efficiency": self.cf_efficiency,
            },
            "ambient": {"model": "boundary"},
        }
        branches = {
            **self._pressurant_branches(),
            "OX_TANK_PUMP": {
                "component": "loss",
                "circuit": "oxidizer",
                "from": "ox_ullage",
                "to": "ox_pump_in",
                "CdA": None,
            },
            "OX_PUMP": {
                "component": "pump",
                "circuit": "oxidizer",
                "from": "ox_pump_in",
                "to": "ox_pump_out",
                "pump_id": self.cfg["pump_ids"]["oxidizer"],
                "CdA": None,
            },
            "OX_PUMP_INJ": {
                "component": "loss",
                "circuit": "oxidizer",
                "from": "ox_pump_out",
                "to": "ox_inj_in" if self.ox_inj_stiffness > 0.0 else "thrust_chamber",
                "CdA": None,
            },
            **({
                "OX_INJ": {
                    "component": "loss",
                    "circuit": "oxidizer",
                    "from": "ox_inj_in",
                    "to": "thrust_chamber",
                    "CdA": None,
                },
            } if self.ox_inj_stiffness > 0.0 else {}),
            "FUEL_TANK_PUMP": {
                "component": "loss",
                "circuit": "fuel",
                "from": "fuel_ullage",
                "to": "fuel_pump_in",
                "CdA": None,
            },
            "FUEL_PUMP": {
                "component": "pump",
                "circuit": "fuel",
                "from": "fuel_pump_in",
                "to": "fuel_pump_out",
                "pump_id": self.cfg["pump_ids"]["fuel"],
                "CdA": None,
            },
            "FUEL_PUMP_INJ": {
                "component": "loss",
                "circuit": "fuel",
                "from": "fuel_pump_out",
                "to": "fuel_inj_in" if self.fuel_inj_stiffness > 0.0 else "thrust_chamber",
                "CdA": None,
            },
            **({
                "FUEL_INJ": {
                    "component": "loss",
                    "circuit": "fuel",
                    "from": "fuel_inj_in",
                    "to": "thrust_chamber",
                    "CdA": None,
                },
            } if self.fuel_inj_stiffness > 0.0 else {}),
            "NOZZLE": {
                "component": "nozzle",
                "circuit": "combustion_gas",
                "from": "thrust_chamber",
                "to": "ambient",
                "At": self.throat_area,
                "Cd": self.nozzle_cd,
            },
        }
        return circuits, nodes, branches

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

        for node_id in nodes:
            incoming = [
                branch["circuit"]
                for branch in branches.values()
                if branch["to"] == node_id
            ]
            if len(incoming) != len(set(incoming)):
                raise ValueError(
                    f"Parallel branches from one circuit cannot enter node '{node_id}'"
                )

        design_flows = {
            "oxidizer": self.mdot_ox,
            "fuel": self.mdot_fuel,
            "exhaust": self.mdot_total,
        }
        def design_flow(branch):
            circuit = circuits[branch["circuit"]]
            flow = branch.get("design_mdot", circuit.get("design_mdot", design_flows.get(circuit["prop"])))
            if flow is None or not isfinite(float(flow)) or float(flow) <= 0:
                raise ValueError(f"Branch {branch} requires a finite positive design_mdot")
            return float(flow)

        for branch_id, branch in branches.items():
            circuit_id = branch["circuit"]
            if circuit_id not in circuits:
                raise ValueError(
                    f"Branch '{branch_id}' references unknown circuit '{circuit_id}'"
                )
            branch_type = branch.get("component", branch.get("model"))
            circuit = circuits[circuit_id]
            branch["fluid"] = circuit["fluid"]
            if (nodes[branch["from"]].get("component") == "pressurant_tank"
                    and branch_type in ("compressible_loss", "bang_bang_valve", "regulator")):
                if branch.get("require_choked", True) is not True:
                    raise ValueError("Pressurant feeds require choked-flow feasibility checks")
                branch["require_choked"] = True
            if circuit["prop"] in design_flows or "design_mdot" in circuit:
                branch.setdefault("design_mdot", design_flow(branch))

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
                sizing = size_electric_pump(definition, design_flow(branch), liquid.rho)
                sizing.update(inlet_pressure_pa=inlet_pressure, inlet_temperature_k=inlet_temperature)
                self.pump_sizing[pump_id] = sizing
                self.sizing_constraints[f"pump.{pump_id}.max_power"] = sizing["power_margin_kw"]
                branch.update(dP=sizing["pressure_rise_pa"], gas_CdA=float(definition["gas_CdA"]))
                if "curve" in definition:
                    curve = scaled_pump_curve(sizing["design_mdot"], sizing["pressure_rise_pa"], **definition["curve"])
                    branch["head_model"] = curve
                    sizing.update(curve_a=curve.a, curve_b=curve.b, curve_c=curve.c,
                                  curve_max_mdot=curve.max_mdot)
                continue

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

            if branch_type in ("bang_bang_valve", "regulator"):
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
                if start_pressure < eol_pressure:
                    tank_id = source_node.get("tank_id", branch["from"])
                    raise DesignInfeasible({f"branch.{branch_id}.choked": start_pressure-eol_pressure,
                                            f"tank.{tank_id}.Pmin": start_pressure-eol_pressure})

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
        """Attach limits to physical tank IDs and choked-flow requirements to edges."""
        self.tank_limits = {}
        self.choked_branches = {}
        initial = dict(self.sizing_constraints)
        for node_id, node in self.node_definitions.items():
            tank_id = node.get("tank_id")
            if tank_id is None:
                continue
            definition = self.tank_definitions.get(tank_id, {})
            limits = {name: float(definition[name]) for name in ("min_temperature", "min_pressure") if name in definition}
            if node.get("component") == "pressurant_tank" and "min_temperature" not in limits:
                legacy = [float(b["min_temperature"]) for b in self.branch_definitions.values()
                          if b["from"] == node_id and "min_temperature" in b]
                if legacy:
                    limits["min_temperature"] = max(legacy)
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
        for branch_id, branch in self.branch_definitions.items():
            required = branch.get("require_choked", False)
            if not isinstance(required, bool):
                raise ValueError("require_choked must be boolean")
            if not required:
                continue
            self.choked_branches[branch_id] = branch
            source = self.node_definitions[branch["from"]]
            target = self.node_definitions[branch["to"]]
            state = source.get("state0", self.circuits[branch["circuit"]]["state0"])
            pressure = float(state["P"])
            gas = self.fluid_properties.state_pt(branch["fluid"], pressure, float(state["T"]))
            downstream = float(target.get("state0", {}).get("P", target.get("P0", 0.0)))
            margin = self._choked_margin(pressure, downstream, gas.gamma)
            initial[f"branch.{branch_id}.choked"] = margin
            if source.get("tank_id") is not None:
                key = f"tank.{source['tank_id']}.Pmin"
                initial[key] = min(initial.get(key, float("inf")), margin)
        self.initial_constraints = initial
        if any(value < 0 for value in initial.values()):
            raise DesignInfeasible(initial, self.pump_sizing)

    @staticmethod
    def _choked_margin(upstream, downstream, gamma):
        if not isfinite(gamma) or gamma <= 1:
            raise ValueError("Choked-flow checking requires finite gamma > 1")
        critical_ratio = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
        return upstream - downstream / critical_ratio

    def _constraint_margins(self, state):
        """Evaluate the same bulk pressures/gas gamma used by compressible flow.

        Closed valves are checked for available choking pressure as well: closing
        a valve must not hide an exhausted supply. Limits apply through shutdown.
        """
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
        for branch_id, branch in self.choked_branches.items():
            source, target = state.nodes[branch["from"]], state.nodes[branch["to"]]
            gas = source.get("fluids", {}).get(branch["fluid"])
            gamma = float(gas["gamma"] if gas is not None else state.branches[branch_id]["gamma"])
            margin = self._choked_margin(float(source["P"]), float(target["P"]), gamma)
            margins[f"branch.{branch_id}.choked"] = margin
            tank_id = self.node_definitions[branch["from"]].get("tank_id")
            if tank_id is not None:
                key = f"tank.{tank_id}.Pmin"
                margins[key] = min(margins.get(key, float("inf")), margin)
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

        result = self.network.update(
            dt=dt,
            bcs=boundaries,
            heat_rate=heat_rate,
            commit=commit,
            axial_specific_force=axial_specific_force,
        )
        return FluidOut(
            node=result["node"],
            branch=result["branch"],
            td_state=result["td_state"],
            mdot=result["mdot"],
            propulsion=self._propulsion_output(result),
            events=result["events"],
            event_counts=result["event_counts"],
            constraints={**self.sizing_constraints, **result["constraints"]},
            constraint_times=result["constraint_times"],
        )
