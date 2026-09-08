from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from .FluidNetwork import FluidNetwork
from .FluidsDef import FluidsDef
from FluidProperties.PropertyModels import (
    CEAPropertySource,
    CombustionPropertySource,
    CoolPropPropertySource,
    PureFluidProperties,
    PureFluidPropertySource,
    TableCombustionPropertySource,
)
from .types import FluidOut, PropulsionOut


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
        self.cfg = cfg["prop_system"]
        self.engine_cfg = cfg["engine"]
        self.model = self.cfg["press_model"]
        self.fluid_properties = (
            fluid_properties
            if fluid_properties is not None
            else CoolPropPropertySource()
        )
        self.combustion_properties = combustion_properties
        tank_geometries = {
            tank_id: tank.get_fluid_geometry() for tank_id, tank in tanks.items()
        }

        self.Pc_target = float(self.cfg["Pc_target"])
        self.MR_target = float(self.cfg["MR_target"])
        self.thrust_target = float(self.cfg["thrust_target"])
        self.fuel_inj_stiffness = float(self.cfg["fuel_inj_stiffness"])
        self.ox_inj_stiffness = float(self.cfg["ox_inj_stiffness"])

        positive = {
            "Pc_target": self.Pc_target,
            "MR_target": self.MR_target,
            "thrust_target": self.thrust_target,
            "fuel_inj_stiffness": self.fuel_inj_stiffness,
            "ox_inj_stiffness": self.ox_inj_stiffness,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"PropSystem inputs must be positive: {invalid}")

        if self.model == "pump_fed":
            self.fuel_pump_head = float(self.cfg["fuel_pump_head"])
            self.ox_pump_head = float(self.cfg["ox_pump_head"])
            self.fuel_inj_pumpout_dp = float(self.cfg["fuel_inj_pumpout_dp"])
            self.ox_inj_pumpout_dp = float(self.cfg["ox_inj_pumpout_dp"])
            self.fuel_pumpin_tank_dp = float(self.cfg["fuel_pumpin_tank_dp"])
            self.ox_pumpin_tank_dp = float(self.cfg["ox_pumpin_tank_dp"])
        elif self.model == "pressure_fed":
            self.fuel_tank_inj_dp = float(self.cfg["fuel_tank_inj_dp"])
            self.ox_tank_inj_dp = float(self.cfg["ox_tank_inj_dp"])
        elif self.model == "blowdown":
            self.fuel_tank_inj_dp = float(self.cfg["fuel_tank_inj_dp"])
            self.ox_tank_inj_dp = float(self.cfg["ox_tank_inj_dp"])
        else:
            raise ValueError(f"Unknown press_model={self.model!r}")

        self._size_engine()
        self.target_ladder = self._build_pressure_ladder()

        circuits, nodes, branches = self._wire_network(tank_geometries)
        self._size_branches(circuits, nodes, branches)
        self.network = FluidNetwork(
            nodes=nodes,
            branches=branches,
            fluid_properties=self.fluid_properties,
            combustion_properties=self.combustion_properties,
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
        self.mdot_total = self.Pc_target * self.throat_area / self.cstar
        self.mdot_ox = self.mdot_total * self.MR_target / (1.0 + self.MR_target)
        self.mdot_fuel = self.mdot_total / (1.0 + self.MR_target)
        self.nozzle_cd = float(self.cfg["nozzle_cd"])

    #users need to modify BPL function along with templates, need to figure a way around this
    def _build_pressure_ladder(self) -> Dict[str, float]:
        Pc = self.Pc_target
        Pox_inj = Pc * (1.0 + self.ox_inj_stiffness)
        Pfuel_inj = Pc * (1.0 + self.fuel_inj_stiffness)
        if self.model == "pump_fed":
            Pox_pump_outlet = Pox_inj + self.ox_inj_pumpout_dp
            Pox_pump_inlet = Pox_pump_outlet - self.ox_pump_head
            Pfuel_pump_outlet = Pfuel_inj + self.fuel_inj_pumpout_dp
            Pfuel_pump_inlet = Pfuel_pump_outlet - self.fuel_pump_head
            return {
                "Pc_target": Pc,
                "Pox_inj": Pox_inj,
                "Pox_pump_outlet": Pox_pump_outlet,
                "Pox_pump_inlet": Pox_pump_inlet,
                "Pox_tank": Pox_pump_inlet + self.ox_pumpin_tank_dp,
                "Pfuel_inj": Pfuel_inj,
                "Pfuel_pump_outlet": Pfuel_pump_outlet,
                "Pfuel_pump_inlet": Pfuel_pump_inlet,
                "Pfuel_tank": Pfuel_pump_inlet + self.fuel_pumpin_tank_dp,
            }
        return {
            "Pc_target": Pc,
            "Pox_inj": Pox_inj,
            "Pox_tank": Pox_inj + self.ox_tank_inj_dp,
            "Pfuel_inj": Pfuel_inj,
            "Pfuel_tank": Pfuel_inj + self.fuel_tank_inj_dp,
        }

    def _state0(self, name: str) -> Dict[str, Any]:
        return dict(self.cfg["state0"][name])

    def _wire_network(
        self,
        tank_geometries: Mapping[str, Any],
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]],
    ]:
        if self.model == "pump_fed":
            return self._template_pump_fed(tank_geometries)
        if self.model == "pressure_fed":
            return self._template_pressure_fed(tank_geometries)
        if self.model == "blowdown":
            return self._template_blowdown(tank_geometries)
        raise ValueError(f"Unknown press_model={self.model!r}")

    def _template_pressure_fed(self, tank_geometries: Mapping[str, Any]):
        if "press_tank" not in tank_geometries:
            raise ValueError("Pressure-fed template requires tank 'press_tank'")
        ox_state0 = self._state0("ox_tank")
        fuel_state0 = self._state0("fuel_tank")
        press_state0 = self._state0("press_tank")
        bang_bang = self.cfg["bang_bang"]
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
                "steady": False,
            },
            "ox_ullage": {
                "component": "propellant_tank",
                "tank_id": "ox_tank",
                "liquid_fluid": ox_fluid,
                "gas_fluid": press_fluid,
                "geometry": tank_geometries["ox_tank"],
                "P0": self.target_ladder["Pox_tank"],
                "state0": ox_state0,
                "steady": False,
            },
            "ox_inj_in": {
                "model": "junction",
                "P0": self.target_ladder["Pox_inj"],
                "steady": True,
            },
            "fuel_ullage": {
                "component": "propellant_tank",
                "tank_id": "fuel_tank",
                "liquid_fluid": fuel_fluid,
                "gas_fluid": press_fluid,
                "geometry": tank_geometries["fuel_tank"],
                "P0": self.target_ladder["Pfuel_tank"],
                "state0": fuel_state0,
                "steady": False,
            },
            "fuel_inj_in": {
                "model": "junction",
                "P0": self.target_ladder["Pfuel_inj"],
                "steady": True,
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
                "steady": True,
            },
            "ambient": {"model": "boundary", "steady": True},
        }
        branches = {
            "OX_BANGBANG": {
                "component": "bang_bang_valve",
                "circuit": "pressurant",
                "from": "press_tank",
                "to": "ox_ullage",
                **bang_bang["OX_BANGBANG"],
                "CdA": None,
            },
            "FUEL_BANGBANG": {
                "component": "bang_bang_valve",
                "circuit": "pressurant",
                "from": "press_tank",
                "to": "fuel_ullage",
                **bang_bang["FUEL_BANGBANG"],
                "CdA": None,
            },
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
                "steady": False,
            },
            "ox_inj_in": {
                "model": "junction",
                "P0": self.target_ladder["Pox_inj"],
                "steady": True,
            },
            "fuel_ullage": {
                "component": "propellant_tank",
                "tank_id": "fuel_tank",
                "liquid_fluid": fuel_fluid,
                "gas_fluid": str(fuel_state0["gas_fluid"]),
                "geometry": tank_geometries["fuel_tank"],
                "P0": self.target_ladder["Pfuel_tank"],
                "state0": fuel_state0,
                "steady": False,
            },
            "fuel_inj_in": {
                "model": "junction",
                "P0": self.target_ladder["Pfuel_inj"],
                "steady": True,
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
                "steady": True,
            },
            "ambient": {"model": "boundary", "steady": True},
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
        if "press_tank" not in tank_geometries:
            raise ValueError("Pump-fed template requires tank 'press_tank'")
        ox_state0 = self._state0("ox_tank")
        fuel_state0 = self._state0("fuel_tank")
        press_state0 = self._state0("press_tank")
        bang_bang = self.cfg["bang_bang"]
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
                "steady": False,
            },
            "ox_ullage": {
                "component": "propellant_tank",
                "tank_id": "ox_tank",
                "liquid_fluid": ox_fluid,
                "gas_fluid": press_fluid,
                "geometry": tank_geometries["ox_tank"],
                "P0": self.target_ladder["Pox_tank"],
                "state0": ox_state0,
                "steady": False,
            },
            "ox_pump_in": {
                "model": "junction",
                "P0": self.target_ladder["Pox_pump_inlet"],
                "steady": True,
            },
            "ox_pump_out": {
                "model": "junction",
                "P0": self.target_ladder["Pox_pump_outlet"],
                "steady": True,
            },
            "ox_inj_in": {
                "model": "junction",
                "P0": self.target_ladder["Pox_inj"],
                "steady": True,
            },
            "fuel_ullage": {
                "component": "propellant_tank",
                "tank_id": "fuel_tank",
                "liquid_fluid": fuel_fluid,
                "gas_fluid": press_fluid,
                "geometry": tank_geometries["fuel_tank"],
                "P0": self.target_ladder["Pfuel_tank"],
                "state0": fuel_state0,
                "steady": False,
            },
            "fuel_pump_in": {
                "model": "junction",
                "P0": self.target_ladder["Pfuel_pump_inlet"],
                "steady": True,
            },
            "fuel_pump_out": {
                "model": "junction",
                "P0": self.target_ladder["Pfuel_pump_outlet"],
                "steady": True,
            },
            "fuel_inj_in": {
                "model": "junction",
                "P0": self.target_ladder["Pfuel_inj"],
                "steady": True,
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
                "steady": True,
            },
            "ambient": {"model": "boundary", "steady": True},
        }
        branches = {
            "OX_BANGBANG": {
                "component": "bang_bang_valve",
                "circuit": "pressurant",
                "from": "press_tank",
                "to": "ox_ullage",
                **bang_bang["OX_BANGBANG"],
                "CdA": None,
            },
            "FUEL_BANGBANG": {
                "component": "bang_bang_valve",
                "circuit": "pressurant",
                "from": "press_tank",
                "to": "fuel_ullage",
                **bang_bang["FUEL_BANGBANG"],
                "CdA": None,
            },
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
                "dP": self.ox_pump_head,
                "CdA": None,
            },
            "OX_PUMP_INJ": {
                "component": "loss",
                "circuit": "oxidizer",
                "from": "ox_pump_out",
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
                "dP": self.fuel_pump_head,
                "CdA": None,
            },
            "FUEL_PUMP_INJ": {
                "component": "loss",
                "circuit": "fuel",
                "from": "fuel_pump_out",
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

        for branch_id, branch in branches.items():
            circuit_id = branch["circuit"]
            if circuit_id not in circuits:
                raise ValueError(
                    f"Branch '{branch_id}' references unknown circuit '{circuit_id}'"
                )
            branch_type = branch.get("component", branch.get("model"))
            circuit = circuits[circuit_id]
            branch["fluid"] = circuit["fluid"]

            if branch_type in ("loss", "incompressible_loss"):
                if branch["CdA"] is not None:
                    continue
                if circuit["prop"] == "oxidizer":
                    mdot = self.mdot_ox
                elif circuit["prop"] == "fuel":
                    mdot = self.mdot_fuel
                else:
                    raise ValueError(
                        f"Liquid circuit '{circuit_id}' must be oxidizer or fuel"
                    )
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
                if circuit["prop"] == "oxidizer":
                    mdot = self.mdot_ox
                elif circuit["prop"] == "fuel":
                    mdot = self.mdot_fuel
                elif circuit["prop"] == "exhaust":
                    mdot = self.mdot_total
                else:
                    raise ValueError(
                        f"Gas circuit '{circuit_id}' has no engine-sized mass flow"
                    )
                branch["CdA"] = FluidsDef.compressible_cda(
                    mdot,
                    upstream_pressure,
                    downstream_pressure,
                    gas.T,
                    gas.R,
                    gas.gamma,
                )
                continue

            if branch_type == "bang_bang_valve":
                target_node = nodes[branch["to"]]
                if target_node.get("component") != "propellant_tank":
                    raise ValueError(
                        f"Bang-bang branch '{branch_id}' must feed a propellant tank"
                    )
                if target_node["gas_fluid"] != circuit["fluid"]:
                    raise ValueError(
                        f"Bang-bang branch '{branch_id}' fluid must match tank ullage"
                    )
                liquid_branches = [
                    candidate
                    for candidate in branches.values()
                    if candidate["from"] == branch["to"]
                    and candidate.get("component", candidate.get("model"))
                    in ("loss", "incompressible_loss")
                ]
                if len(liquid_branches) != 1:
                    raise ValueError(
                        f"Tank '{branch['to']}' requires one liquid outlet"
                    )
                liquid_circuit = liquid_branches[0]["circuit"]
                liquid_prop = circuits[liquid_circuit]["prop"]
                if liquid_prop == "oxidizer":
                    liquid_mdot = self.mdot_ox
                elif liquid_prop == "fuel":
                    liquid_mdot = self.mdot_fuel
                else:
                    raise ValueError(
                        f"Tank '{branch['to']}' outlet must be oxidizer or fuel"
                    )

                source_node = nodes[branch["from"]]
                if source_node.get("component") != "pressurant_tank":
                    raise ValueError(
                        f"Bang-bang branch '{branch_id}' requires a gas-volume source"
                    )
                if source_node["fluid"] != circuit["fluid"]:
                    raise ValueError(
                        f"Bang-bang branch '{branch_id}' fluid must match its source"
                    )
                initial_gas = circuit_properties(circuit_id)
                downstream_pressure = float(target_node["P0"])
                start_pressure = float(nodes[branch["from"]]["P0"])
                if downstream_pressure <= 0.0 or start_pressure <= 0.0:
                    raise ValueError(
                        f"Bang-bang branch '{branch_id}' pressures must be positive"
                    )
                start_temperature = initial_gas.T
                gamma = initial_gas.gamma
                critical_ratio = (2.0 / (gamma + 1.0)) ** (
                    gamma / (gamma - 1.0)
                )
                eol_pressure = downstream_pressure / critical_ratio
                if start_pressure <= eol_pressure:
                    raise ValueError(
                        f"Bang-bang branch '{branch_id}' cannot remain choked at EOL"
                    )

                pressure_mid = 0.5 * (start_pressure + eol_pressure)
                min_temperature = float(branch["min_temperature"])
                collapse_factor = float(branch["collapse_factor"])
                if min_temperature <= 0.0 or collapse_factor <= 0.0:
                    raise ValueError(
                        f"Bang-bang branch '{branch_id}' sizing inputs must be positive"
                    )
                temperature_mid = 0.5 * (start_temperature + min_temperature)
                gas = self.fluid_properties.state_pt(
                    circuit["fluid"],
                    pressure_mid,
                    temperature_mid,
                )
                gamma = gas.gamma
                gas_constant = gas.R
                duty_cycle = float(branch["duty_cycle"])
                if not 0.0 < duty_cycle <= 1.0:
                    raise ValueError(
                        f"Gas branch '{branch_id}' duty cycle must be in (0, 1]"
                    )

                liquid_vdot = liquid_mdot / circuit_properties(liquid_circuit).rho
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
                branch["design_pressure"] = pressure_mid
                branch["design_temperature"] = temperature_mid
                branch["duty_cycle"] = duty_cycle
                branch["eol_pressure"] = eol_pressure

    def _propulsion_output(
        self,
        network_output: Dict[str, Any],
    ) -> PropulsionOut:
        mdot_ox = network_output["mdot"]["OX_INJ"]
        mdot_fuel = network_output["mdot"]["FUEL_INJ"]
        chamber = network_output["node"]["thrust_chamber"]
        Pc = chamber["P"]
        MR = chamber["MR"]
        Cf = chamber["Cf"]
        mode = chamber["mode"]
        nozzle = network_output["branch"]["NOZZLE"]
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
            mdot_nozzle=network_output["mdot"]["NOZZLE"],
        )

    def update(
        self,
        dt: Optional[float],
        atm: Any,
        heat_flux: Dict[str, float],
        bcs: Optional[Dict[str, Dict[str, Any]]] = None,
        commit: bool = True,
    ) -> FluidOut:
        boundaries = dict(bcs or {})
        boundaries["ambient"] = {"P": float(atm.p)}

        result = self.network.update(
            dt=dt,
            bcs=boundaries,
            heat_flux=heat_flux,
            commit=commit,
        )
        return FluidOut(
            node=result["node"],
            branch=result["branch"],
            td_state=result["td_state"],
            mdot=result["mdot"],
            propulsion=self._propulsion_output(result),
        )
