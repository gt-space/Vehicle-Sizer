from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .FluidsDef import FluidsDef
from FluidProperties.PropertyModels import (
    CombustionPropertySource,
    PureFluidPropertySource,
)


@dataclass(frozen=True)
class FlowConn:
    """Runtime flow and constituent states at one node/branch connection."""

    id: str
    sign: float
    state: Dict[str, Any]

    @property
    def components(self) -> Dict[str, Dict[str, Any]]:
        return self.state["components"]

    @property
    def enabled(self) -> bool:
        return bool(self.state["enabled"])

    @property
    def fluid(self) -> str:
        if len(self.components) != 1:
            raise ValueError(f"Flow connection '{self.id}' contains multiple fluids")
        return next(iter(self.components))

    @property
    def phase(self) -> str:
        return str(self.components[self.fluid]["phase"])

    @property
    def mdot(self) -> float:
        return sum(float(item["mdot"]) for item in self.components.values())

    @property
    def h(self) -> float:
        return float(self.components[self.fluid]["h"])

    def mdot_of(self, fluid: str) -> float:
        item = self.components.get(fluid)
        return 0.0 if item is None else float(item["mdot"])

    def phase_mdot(self, phase: str) -> float:
        return sum(
            float(item["mdot"])
            for item in self.components.values()
            if item["phase"] == phase
        )


class NodeModel:
    """Physics interface used by a fluid node."""

    is_dynamic = False
    flow_coupled = False
    routes_inlet = False

    def initial_state(self, node) -> Dict[str, Any]:
        return {}

    def state_variables(self, node, dt, prescribed) -> Dict[str, float]:
        return {}

    def trial_state(self, node, state, adjacent, node_states) -> Dict[str, Any]:
        return dict(state)

    def residual(
        self, node, trial, previous, evaluated, adjacent, dt, heat_flux
    ) -> np.ndarray:
        return np.empty(0)

    def outlet_state(
        self, node, evaluated, adjacent, preferred_fluid=None
    ) -> Dict[str, Dict[str, Any]]:
        fluids = evaluated.get("fluids", {})
        if preferred_fluid in fluids:
            fluids = {preferred_fluid: fluids[preferred_fluid]}
        if len(fluids) != 1:
            raise ValueError(f"Node '{node.id}' does not expose one fluid stream")
        return {fluid: dict(state) for fluid, state in fluids.items()}


class SteadyModel(NodeModel):
    """Base for zero-capacitance nodes whose pressure is algebraic."""

    def initial_state(self, node) -> Dict[str, float]:
        return {"P": float(node.definition["P0"])}

    def state_variables(self, node, dt, prescribed) -> Dict[str, float]:
        return {} if prescribed is not None else {"P": float(node.state["P"])}


class JunctionModel(SteadyModel):
    """Single-stream algebraic junction."""

    routes_inlet = True

    def residual(
        self, node, trial, previous, evaluated, adjacent, dt, heat_flux
    ) -> np.ndarray:
        return np.array(
            [sum(item.sign * item.mdot for item in adjacent if item.enabled)],
            dtype=float,
        )

    def outlet_state(
        self, node, evaluated, adjacent, preferred_fluid=None
    ) -> Dict[str, Dict[str, Any]]:
        components = {
            fluid: state
            for item in adjacent
            if item.enabled and item.sign * item.mdot >= 0.0
            for fluid, state in item.components.items()
        }
        if len(components) != 1:
            raise ValueError(
                f"Junction '{node.id}' requires one incoming fluid, got "
                f"{sorted(components)}"
            )
        return {fluid: dict(state) for fluid, state in components.items()}


class BoundaryModel(NodeModel):
    """Prescribed node with no solver unknowns."""

    def initial_state(self, node) -> Dict[str, Any]:
        return dict(node.definition)

    def trial_state(self, node, state, adjacent, node_states) -> Dict[str, Any]:
        if "P" not in state:
            raise ValueError(f"Boundary node '{node.id}' requires a pressure state")
        return dict(state)


class DynamicModel(NodeModel):
    is_dynamic = True

    def state_variables(self, node, dt, prescribed) -> Dict[str, float]:
        if dt is None or prescribed is not None:
            return {}
        return dict(node.state)

    @staticmethod
    def fluxes(
        adjacent: List[FlowConn], fluid: Optional[str] = None
    ) -> Tuple[float, float]:
        mdot = 0.0
        hdot = 0.0
        for item in adjacent:
            if not item.enabled:
                continue
            for name, component in item.components.items():
                if fluid is not None and name != fluid:
                    continue
                signed_mdot = item.sign * float(component["mdot"])
                mdot += signed_mdot
                hdot += signed_mdot * float(component["h"])
        return mdot, hdot


class VolumeModel(DynamicModel):
    """Finite-capacitance single-phase volume."""

    def initial_state(self, node) -> Dict[str, float]:
        state0 = node.definition["state0"]
        return {"m": float(state0["m"]), "U": float(state0["U"])}

    def trial_state(self, node, state, adjacent, node_states) -> Dict[str, Any]:
        geometry = node.definition["geometry"]
        fluid = node.definition["fluid"]
        properties = node.require_fluid_properties().state_rho_u(
            fluid,
            state["m"] / geometry.volume,
            state["U"] / state["m"],
        ).as_dict()
        properties["V"] = geometry.volume
        properties["phase"] = node.fluid_phase(fluid)
        return {
            **state,
            "mass": state["m"],
            "tank_id": node.definition.get("tank_id"),
            "axial_mass": node.axial_mass(state["m"]),
            "P": properties["P"],
            **node.volume_metadata(),
            "fluids": {fluid: properties},
        }

    def residual(
        self, node, trial, previous, evaluated, adjacent, dt, heat_flux
    ) -> np.ndarray:
        if dt is None:
            return np.empty(0)
        mdot, hdot = self.fluxes(adjacent)
        qdot = float(heat_flux.get(node.id, 0.0)) * float(
            node.definition["geometry"].internal_area
        )
        mass_scale = max(abs(previous["m"]), 1.0)
        energy_scale = max(abs(previous["U"]), 1.0)
        return np.array(
            [
                (trial["m"] - previous["m"] - dt * mdot) / mass_scale,
                (trial["U"] - previous["U"] - dt * (hdot + qdot))
                / energy_scale,
            ],
            dtype=float,
        )


class TwoPhaseVolumeModel(DynamicModel):
    """Finite-capacitance liquid inventory with a gas ullage."""

    names = ("m_liq", "U_liq", "m_ull", "U_ull")

    def initial_state(self, node) -> Dict[str, float]:
        state0 = node.definition["state0"]
        return {name: float(state0[name]) for name in self.names}

    def trial_state(self, node, state, adjacent, node_states) -> Dict[str, Any]:
        definition = node.definition
        geometry = definition["geometry"]
        liquid_fluid = definition["liquid_fluid"]
        gas_fluid = definition["gas_fluid"]
        try:
            tank = FluidsDef.tank_compatibility(
                m_liquid=state["m_liq"],
                U_liquid=state["U_liq"],
                m_gas=state["m_ull"],
                U_gas=state["U_ull"],
                tank_volume=geometry.volume,
                liquid_fluid=liquid_fluid,
                gas_fluid=gas_fluid,
                pressure_guess=definition["P0"],
                fluid_properties=node.require_fluid_properties(),
            )
        except (ValueError, RuntimeError, ZeroDivisionError):
            if node.evaluated is None:
                raise
            return {
                **node.evaluated,
                **state,
                "mass": state["m_liq"] + state["m_ull"],
                "_trial_valid": False,
            }
        fill = geometry.fill_state(tank["liquid"]["V"])
        return {
            **state,
            "mass": state["m_liq"] + state["m_ull"],
            "tank_id": definition.get("tank_id"),
            "mode": node.mode,
            "axial_mass": geometry.axial_mass(
                liquid_volume=tank["liquid"]["V"],
                liquid_mass=state["m_liq"],
                ullage_mass=state["m_ull"],
            ),
            "P": tank["P"],
            "_trial_valid": True,
            **fill,
            "fluids": {
                liquid_fluid: {
                    **tank["liquid"],
                    "phase": "liquid",
                    "contact_area": fill["liquid_contact_area"],
                },
                gas_fluid: {
                    **tank["gas"],
                    "phase": "gas",
                    "contact_area": fill["ullage_contact_area"],
                },
            },
        }

    def residual(
        self, node, trial, previous, evaluated, adjacent, dt, heat_flux
    ) -> np.ndarray:
        if dt is None:
            return np.empty(0)
        liquid = node.definition["liquid_fluid"]
        gas = node.definition["gas_fluid"]
        mdot_liq, hdot_liq = self.fluxes(adjacent, liquid)
        mdot_gas, hdot_gas = self.fluxes(adjacent, gas)
        q_flux = float(heat_flux.get(node.id, 0.0))
        qdot_liq = q_flux * evaluated["fluids"][liquid]["contact_area"]
        qdot_gas = q_flux * evaluated["fluids"][gas]["contact_area"]
        if node.evaluated is None:
            raise RuntimeError(f"Node '{node.id}' has no previous evaluated state")
        pressure = evaluated["P"]
        dV_liq = (
            evaluated["fluids"][liquid]["V"]
            - node.evaluated["fluids"][liquid]["V"]
        )
        dV_gas = (
            evaluated["fluids"][gas]["V"]
            - node.evaluated["fluids"][gas]["V"]
        )
        liquid_mass_scale = max(abs(previous["m_liq"]), 1.0)
        liquid_energy_scale = max(abs(previous["U_liq"]), 1.0)
        gas_mass_scale = max(abs(previous["m_ull"]), 1.0)
        gas_energy_scale = max(abs(previous["U_ull"]), 1.0)
        return np.array(
            [
                (trial["m_liq"] - previous["m_liq"] - dt * mdot_liq)
                / liquid_mass_scale,
                (
                    trial["U_liq"]
                    - previous["U_liq"]
                    - dt * (hdot_liq + qdot_liq)
                    + pressure * dV_liq
                )
                / liquid_energy_scale,
                (trial["m_ull"] - previous["m_ull"] - dt * mdot_gas)
                / gas_mass_scale,
                (
                    trial["U_ull"]
                    - previous["U_ull"]
                    - dt * (hdot_gas + qdot_gas)
                    + pressure * dV_gas
                )
                / gas_energy_scale,
            ],
            dtype=float,
        )


class CombustionModel(SteadyModel):
    """Algebraic reacting chamber."""

    flow_coupled = True

    def trial_state(self, node, state, adjacent, node_states) -> Dict[str, Any]:
        definition = node.definition

        def inflow(fluid: str) -> float:
            return sum(
                item.sign * item.mdot_of(fluid)
                for item in adjacent
                if item.enabled
            )

        mdot_ox = inflow(definition["oxidizer_fluid"])
        mdot_fuel = inflow(definition["fuel_fluid"])
        flowing = mdot_ox > 0.0 and mdot_fuel > 0.0
        mixture_ratio = mdot_ox / mdot_fuel if flowing else 0.0
        if flowing:
            product = node.require_combustion_properties().evaluate(
                chamber_pressure=float(state["P"]),
                mixture_ratio=mixture_ratio,
                ambient_pressure=node_states[definition["ambient_node"]]["P"],
                expansion_ratio=definition["expansion_ratio"],
                cstar_efficiency=definition["cstar_efficiency"],
                cf_efficiency=definition["cf_efficiency"],
            ).as_dict()
        else:
            product = {
                name: 0.0 for name in ("cstar", "Cf", "R", "gamma", "T")
            }
        combustion_fluid = definition["combustion_fluid"]
        return {
            **state,
            "cstar": product["cstar"],
            "Cf": product["Cf"],
            "MR": mixture_ratio,
            "mdot_oxidizer": mdot_ox,
            "mdot_fuel": mdot_fuel,
            "mode": node.mode,
            "shutdown_reason": node.shutdown_reason,
            "fluids": {
                combustion_fluid: {
                    **{name: product[name] for name in ("R", "gamma", "T")},
                    "phase": "gas",
                }
            },
        }

    def residual(
        self, node, trial, previous, evaluated, adjacent, dt, heat_flux
    ) -> np.ndarray:
        return np.array(
            [sum(item.sign * item.mdot for item in adjacent if item.enabled)],
            dtype=float,
        )


class TwinPathJunctionModel(SteadyModel):
    """Shutdown chamber conserving gas and liquid on separate paths."""

    flow_coupled = True

    def __init__(self, phases):
        self.phases = tuple(phases)

    def state_variables(self, node, dt, prescribed) -> Dict[str, float]:
        return super().state_variables(node, dt, prescribed) if self.phases else {}

    def trial_state(self, node, state, adjacent, node_states) -> Dict[str, Any]:
        if not self.phases:
            state = {
                **state,
                "P": node_states[node.definition["ambient_node"]]["P"],
            }
        sources: Dict[str, Dict[str, Any]] = {}
        for item in adjacent:
            if not item.enabled or item.sign <= 0.0:
                continue
            for fluid, component in item.components.items():
                if component["phase"] in self.phases:
                    sources[fluid] = component

        fluids = {}
        for fluid, source in sources.items():
            props = dict(source)
            if "T" in source:
                props = node.require_fluid_properties().state_pt(
                    fluid, float(state["P"]), float(source["T"])
                ).as_dict()
            fluids[fluid] = {
                **props,
                "phase": source["phase"],
                "mdot": sum(
                    item.sign * item.mdot_of(fluid)
                    for item in adjacent
                    if item.enabled and item.sign > 0.0
                ),
            }
        return {
            **state,
            "cstar": 0.0,
            "Cf": 0.0,
            "MR": 0.0,
            "mode": node.mode,
            "shutdown_reason": node.shutdown_reason,
            "fluids": fluids,
        }

    def residual(
        self, node, trial, previous, evaluated, adjacent, dt, heat_flux
    ) -> np.ndarray:
        return np.array(
            [
                sum(
                    item.sign * item.phase_mdot(phase)
                    for item in adjacent
                    if item.enabled
                )
                for phase in self.phases
            ],
            dtype=float,
        )

    def outlet_state(
        self, node, evaluated, adjacent, preferred_fluid=None
    ) -> Dict[str, Dict[str, Any]]:
        fluids = evaluated.get("fluids", {})
        if not fluids and self.phases:
            raise ValueError(f"Twin-path junction '{node.id}' has no inlet streams")
        return {fluid: dict(state) for fluid, state in fluids.items()}


class FluidNode:
    """Solver node holding state, connectivity, and an active physics model."""

    def __init__(self, node_id: str, definition: Dict[str, Any], model: NodeModel):
        self.id = node_id
        self.definition = definition
        self.model = model
        self.incoming: List[str] = []
        self.outgoing: List[str] = []
        self.state: Dict[str, Any] = self.model.initial_state(self)
        self.evaluated: Optional[Dict[str, Any]] = None
        self.fluid_properties: Optional[PureFluidPropertySource] = None
        self.combustion_properties: Optional[CombustionPropertySource] = None

    @property
    def is_dynamic(self) -> bool:
        return self.model.is_dynamic

    def switch_model(self, model: NodeModel, state: Dict[str, Any]) -> None:
        self.model = model
        self.state = dict(state)

    def state_variables(self, dt, prescribed) -> Dict[str, float]:
        return self.model.state_variables(self, dt, prescribed)

    def trial_state(self, state, adjacent, node_states) -> Dict[str, Any]:
        return self.model.trial_state(self, state, adjacent, node_states)

    def residual(
        self, trial, previous, evaluated, adjacent, dt, heat_flux
    ) -> np.ndarray:
        return self.model.residual(
            self, trial, previous, evaluated, adjacent, dt, heat_flux
        )

    def commit(
        self,
        state: Dict[str, Any],
        evaluated: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.state = dict(state)
        if evaluated is not None:
            self.evaluated = dict(evaluated)

    def axial_mass(self, mass: float):
        return self.definition["geometry"].axial_mass(mass)

    def volume_metadata(self) -> Dict[str, Any]:
        return {}

    def fluid_phase(self, fluid: str) -> str:
        phase = self.definition.get("phase")
        if phase not in ("liquid", "gas"):
            raise ValueError(
                f"Node '{self.id}' requires phase='liquid' or phase='gas'"
            )
        return str(phase)

    def require_fluid_properties(self) -> PureFluidPropertySource:
        if self.fluid_properties is None:
            raise ValueError(f"Node '{self.id}' requires a fluid-property source")
        return self.fluid_properties

    def require_combustion_properties(self) -> CombustionPropertySource:
        if self.combustion_properties is None:
            raise ValueError(f"Node '{self.id}' requires a combustion-property source")
        return self.combustion_properties


class PressurantTankComponent(FluidNode):
    def __init__(self, node_id: str, definition: Dict[str, Any]):
        super().__init__(node_id, definition, VolumeModel())

    def fluid_phase(self, fluid: str) -> str:
        return "gas"


class PropellantTankComponent(FluidNode):
    dry_fraction = 1.0e-4
    max_dry_mass = 1.0e-4

    def __init__(self, node_id: str, definition: Dict[str, Any]):
        self.mode = "two_phase"
        initial_liquid_mass = float(definition["state0"]["m_liq"])
        self.dry_mass = max(
            min(initial_liquid_mass * self.dry_fraction, self.max_dry_mass),
            1.0e-12,
        )
        super().__init__(node_id, definition, TwoPhaseVolumeModel())
        if initial_liquid_mass <= 0.0:
            self.dry_out()

    def time_to_dry(self, adjacent: List[FlowConn]) -> Optional[float]:
        if self.mode != "two_phase":
            return None
        liquid = self.definition["liquid_fluid"]
        mdot = sum(
            item.sign * item.mdot_of(liquid)
            for item in adjacent
            if item.enabled
        )
        if mdot >= 0.0:
            return None
        return max((self.state["m_liq"] - self.dry_mass) / -mdot, 0.0)

    def dry_out(self) -> bool:
        if self.mode != "two_phase":
            return False
        if self.state["m_liq"] > self.dry_mass:
            return False
        self.mode = "gas"
        self.definition["fluid"] = self.definition["gas_fluid"]
        self.switch_model(
            VolumeModel(),
            {"m": self.state["m_ull"], "U": self.state["U_ull"]},
        )
        return True

    def axial_mass(self, mass: float):
        return self.definition["geometry"].axial_mass(
            liquid_volume=0.0,
            liquid_mass=0.0,
            ullage_mass=mass,
        )

    def volume_metadata(self) -> Dict[str, Any]:
        return {"mode": self.mode, **self.definition["geometry"].fill_state(0.0)}

    def fluid_phase(self, fluid: str) -> str:
        return "gas" if self.mode == "gas" else super().fluid_phase(fluid)


class CombustorComponent(FluidNode):
    def __init__(self, node_id: str, definition: Dict[str, Any]):
        self.mode = "combusting"
        self.shutdown_reason: Optional[str] = None
        super().__init__(node_id, definition, CombustionModel())

    def update_mode(self, inflows: List[FlowConn]) -> bool:
        phases = tuple(
            sorted(
                {
                    component["phase"]
                    for item in inflows
                    if item.enabled and item.sign > 0.0
                    for component in item.components.values()
                }
            )
        )
        if self.mode == "shutdown":
            if getattr(self.model, "phases", ()) == phases:
                return False
            self.switch_model(TwinPathJunctionModel(phases), self.state)
            return True
        flow = {
            fluid: sum(
                item.sign * item.mdot_of(fluid)
                for item in inflows
                if item.enabled and item.sign > 0.0
            )
            for item in inflows
            for fluid in item.components
        }
        missing_ox = flow.get(self.definition["oxidizer_fluid"], 0.0) <= 0.0
        missing_fuel = flow.get(self.definition["fuel_fluid"], 0.0) <= 0.0
        if not (missing_ox or missing_fuel):
            return False
        if missing_ox and missing_fuel:
            self.shutdown_reason = "propellants_unavailable"
        elif missing_ox:
            self.shutdown_reason = "oxidizer_unavailable"
        else:
            self.shutdown_reason = "fuel_unavailable"
        self.mode = "shutdown"
        state = {
            **self.state,
            "cstar": 0.0,
            "Cf": 0.0,
            "MR": 0.0,
            "mdot_oxidizer": 0.0,
            "mdot_fuel": 0.0,
            "mode": self.mode,
            "shutdown_reason": self.shutdown_reason,
        }
        self.switch_model(TwinPathJunctionModel(phases), state)
        return True
