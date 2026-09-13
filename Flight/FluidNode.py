from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from FluidProperties.PropertyModels import (
    CombustionPropertySource,
    PureFluidPropertySource,
)
from .FluidState import BranchState, FluidState, NodeState


Adjacent = List[Tuple[float, BranchState]]


def iter_flows(adjacent: Adjacent):
    for incidence, branch in adjacent:
        if branch.enabled:
            for flow in branch.flows.values():
                yield incidence, flow


def net_mdot(adjacent: Adjacent, fluid: Optional[str] = None, phase: Optional[str] = None):
    return sum(
        incidence * float(flow["mdot"])
        for incidence, flow in iter_flows(adjacent)
        if (fluid is None or flow["fluid"].fluid == fluid)
        and (phase is None or flow["fluid"].phase == phase)
    )


class NodeModel:
    """Physics interface used by a fluid node."""

    is_dynamic = False
    flow_coupled = False
    routes_inlet = False

    def initial_state(self, node) -> Dict[str, Any]:
        return {}

    def state(self, node, dt, prescribed) -> Dict[str, float]:
        return {}

    def scales(self, node, state) -> Dict[str, float]:
        return {
            name: max(abs(float(value)), 1.0)
            for name, value in state.items()
        }

    def committed_state(self, node, state, evaluated) -> Dict[str, float]:
        return dict(state)

    def output_state(self, node, evaluated) -> Dict[str, Any]:
        """Enrich one converged node state for external consumers."""

        return dict(evaluated)

    def trial_state(self, node, state, adjacent, node_states) -> Dict[str, Any]:
        return dict(state)

    def residual(
        self, node, trial, previous, evaluated, adjacent, dt, heat_flux
    ) -> np.ndarray:
        return np.empty(0)

    def outlet(self, node, evaluated, adjacent, port=None) -> Dict[str, FluidState]:
        fluids = evaluated.get("fluids", {})
        selected = node.fluid_for_port(port)
        if selected in fluids:
            fluids = {selected: fluids[selected]}
        if len(fluids) != 1:
            raise ValueError(f"Node '{node.id}' does not expose one fluid stream")
        return {name: fluid if isinstance(fluid, FluidState) else FluidState.from_dict(name, fluid) for name, fluid in fluids.items()}


class SteadyModel(NodeModel):
    """Base for zero-capacitance nodes whose pressure is algebraic."""

    def initial_state(self, node) -> Dict[str, float]:
        return {"P": float(node.definition["P0"])}

    def state(self, node, dt, prescribed) -> Dict[str, float]:
        return {} if prescribed is not None else {"P": float(node.state["P"])}

    def scales(self, node, variables) -> Dict[str, float]:
        return {"P": max(abs(float(node.definition["P0"])), 1.0)}

    @staticmethod
    def continuity_scale(adjacent: Adjacent) -> float:
        return max(sum(abs(float(flow["mdot"])) for _, flow in iter_flows(adjacent)), 1.0)


class JunctionModel(SteadyModel):
    """Single-stream algebraic junction."""

    routes_inlet = True

    def residual(
        self, node, trial, previous, evaluated, adjacent, dt, heat_flux
    ) -> np.ndarray:
        return np.array(
            [
                net_mdot(adjacent)
                / self.continuity_scale(adjacent)
            ],
            dtype=float,
        )

    def outlet(self, node, evaluated, adjacent, port=None) -> Dict[str, FluidState]:
        fluids = {
            flow["fluid"].fluid: flow["fluid"]
            for incidence, flow in iter_flows(adjacent)
            if incidence * int(flow["direction"]) > 0
        }
        if len(fluids) != 1:
            raise ValueError(
                f"Junction '{node.id}' requires one incoming fluid, got "
                f"{sorted(fluids)}"
            )
        return fluids


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

    @staticmethod
    def fluxes(
        adjacent: Adjacent, fluid: Optional[str] = None
    ) -> Tuple[float, float]:
        mdot = 0.0
        hdot = 0.0
        for incidence, flow in iter_flows(adjacent):
            state = flow["fluid"]
            if fluid is not None and state.fluid != fluid:
                continue
            signed_mdot = incidence * float(flow["mdot"])
            mdot += signed_mdot
            hdot += signed_mdot * float(state["h"])
        return mdot, hdot


class VolumeModel(DynamicModel):
    """Finite volume supporting single-phase and saturated pure-fluid states."""

    def initial_state(self, node) -> Dict[str, float]:
        state0 = node.definition["state0"]
        return {
            name: float(state0[name])
            for name in ("P", "T", "m", "U")
        }

    def state(self, node, dt, prescribed) -> Dict[str, float]:
        if dt is None or prescribed is not None:
            return {}
        if "quality" in node.state:
            return {
                name: float(node.state[name])
                for name in ("P", "quality", "m")
            }
        return {name: float(node.state[name]) for name in ("P", "T")}

    def scales(self, node, variables) -> Dict[str, float]:
        state0 = node.definition["state0"]
        if "quality" in variables:
            return {
                "P": max(abs(float(state0.get("P", node.state["P"]))), 1.0),
                "quality": 1.0,
                "m": max(abs(float(state0.get("m", node.state["m"]))), 1.0e-6),
            }
        return {
            "P": max(abs(float(state0.get("P", node.state["P"]))), 1.0),
            "T": max(abs(float(state0.get("T", node.state["T"]))), 1.0),
        }

    def trial_state(self, node, state, adjacent, node_states) -> Dict[str, Any]:
        geometry = node.definition["geometry"]
        fluid = node.definition["fluid"]
        if "quality" in state:
            lower, upper = node.require_fluid_properties().saturation_bounds(fluid)
            saturation = node.require_fluid_properties().saturation_at_p(
                fluid, min(max(state["P"], lower), upper)
            )
            quality = float(state["quality"])
            mass = float(state["m"])
            specific_volume = (
                (1.0 - quality) / saturation.liquid.rho
                + quality / saturation.vapor.rho
            )
            specific_energy = (
                (1.0 - quality) * saturation.liquid.u
                + quality * saturation.vapor.u
            )
            properties = saturation.vapor.as_dict()
            properties.update(
                {
                    "V": quality * mass / saturation.vapor.rho,
                    "phase": "gas",
                }
            )
            energy = mass * specific_energy
            occupied_volume = mass * specific_volume
            temperature = saturation.T
        elif (
            not getattr(node, "_condensation_armed", True)
            and node.evaluated is not None
            and fluid in node.evaluated.fluids
        ):
            reference = node.evaluated.fluids[fluid]
            temperature = float(state["T"])
            pressure = float(state["P"])
            cv = reference["R"] / (reference["gamma"] - 1.0)
            cp = cv + reference["R"]
            properties = {
                "P": pressure,
                "T": temperature,
                "rho": reference["rho"]
                * pressure / reference["P"]
                * reference["T"] / temperature,
                "h": reference["h"] + cp * (temperature - reference["T"]),
                "u": reference["u"] + cv * (temperature - reference["T"]),
                "R": reference["R"],
                "gamma": reference["gamma"],
                "V": geometry.volume,
                "phase": "gas",
            }
            mass = properties["rho"] * geometry.volume
            energy = mass * properties["u"]
            occupied_volume = geometry.volume
        else:
            properties = node.require_fluid_properties().state_pt(
                fluid,
                state["P"],
                state["T"],
            ).as_dict()
            properties["V"] = geometry.volume
            properties["phase"] = node.fluid_phase(fluid)
            mass = properties["rho"] * geometry.volume
            energy = mass * properties["u"]
            occupied_volume = geometry.volume
            temperature = properties["T"]
        return {
            **state,
            "T": temperature,
            "m": mass,
            "U": energy,
            "quality": state.get("quality"),
            "occupied_volume": occupied_volume,
            "P": state["P"],
            "fluids": {fluid: properties},
        }

    def output_state(self, node, evaluated) -> Dict[str, Any]:
        fluid = node.definition["fluid"]
        output = {
            **evaluated,
            "mass": evaluated["m"],
            "tank_id": node.definition.get("tank_id"),
            "axial_mass": node.axial_mass(evaluated["m"]),
            **node.volume_metadata(),
        }
        if evaluated.get("quality") is None:
            properties = evaluated["fluids"][fluid]
            output["phase_states"] = {properties["phase"]: dict(properties)}
        else:
            lower, upper = node.require_fluid_properties().saturation_bounds(fluid)
            saturation = node.require_fluid_properties().saturation_at_p(
                fluid, min(max(evaluated["P"], lower), upper)
            )
            output["phase_states"] = {
                "liquid": saturation.liquid.as_dict(),
                "gas": saturation.vapor.as_dict(),
            }
        return output

    def committed_state(self, node, state, evaluated) -> Dict[str, float]:
        names = (
            ("P", "T", "quality", "m", "U")
            if evaluated.get("quality") is not None
            else ("P", "T", "m", "U")
        )
        return {name: float(evaluated[name]) for name in names}

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
        equations = [
            (evaluated["m"] - previous["m"] - dt * mdot) / mass_scale,
            (evaluated["U"] - previous["U"] - dt * (hdot + qdot))
            / energy_scale,
        ]
        if "quality" in trial:
            equations.append(
                (
                    evaluated["occupied_volume"]
                    - node.definition["geometry"].volume
                )
                / node.definition["geometry"].volume
            )
        return np.asarray(equations, dtype=float)

    def condense(self, node) -> bool:
        """Enter saturated equilibrium when a gas volume reaches its dew line."""

        if (
            "quality" in node.state
            or node.fluid_phase(node.definition["fluid"]) != "gas"
        ):
            return False
        fluid = node.definition["fluid"]
        if not node.require_fluid_properties().supports_saturation(fluid):
            return False
        pressure = float(node.state["P"])
        lower, upper = node.require_fluid_properties().saturation_bounds(fluid)
        if not lower <= pressure <= upper:
            return False
        saturation = node.require_fluid_properties().saturation_at_p(fluid, pressure)
        if float(node.state["T"]) > saturation.T:
            return False
        volume = node.definition["geometry"].volume
        specific_volume = volume / float(node.state["m"])
        quality = (
            specific_volume - 1.0 / saturation.liquid.rho
        ) / (1.0 / saturation.vapor.rho - 1.0 / saturation.liquid.rho)
        if quality > 1.01:
            return False
        node.state = {
            "P": pressure,
            "T": saturation.T,
            "quality": min(max(quality, 0.0), 1.0),
            "m": float(node.state["m"]),
            "U": float(node.state["U"]),
        }
        return True


class TwoSpeciesModel(DynamicModel):
    """Finite-capacitance liquid inventory with a gas ullage."""

    names = ("P", "T_liq", "T_ull", "m_liq", "m_ull")

    def initial_state(self, node) -> Dict[str, float]:
        state0 = node.definition["state0"]
        return {
            "P": float(state0["P"]),
            "T_liq": float(state0["T"]),
            "T_ull": float(state0["gas_T"]),
            "m_liq": float(state0["m_liq"]),
            "U_liq": float(state0["U_liq"]),
            "m_ull": float(state0["m_ull"]),
            "U_ull": float(state0["U_ull"]),
        }

    def state(self, node, dt, prescribed) -> Dict[str, float]:
        if dt is None or prescribed is not None:
            return {}
        return {name: float(node.state[name]) for name in self.names}

    def scales(self, node, variables) -> Dict[str, float]:
        state0 = node.definition["state0"]
        return {
            "P": max(abs(float(state0["P"])), 1.0),
            "T_liq": max(abs(float(state0["T"])), 1.0),
            "T_ull": max(abs(float(state0["gas_T"])), 1.0),
            "m_liq": max(abs(float(state0["m_liq"])), 1.0),
            "m_ull": max(abs(float(state0["m_ull"])), 1.0e-6),
        }

    def trial_state(self, node, state, adjacent, node_states) -> Dict[str, Any]:
        definition = node.definition
        geometry = definition["geometry"]
        liquid_fluid = definition["liquid_fluid"]
        gas_fluid = definition["gas_fluid"]
        pressure = state["P"]
        liquid = node.require_fluid_properties().state_pt(
            liquid_fluid, pressure, state["T_liq"]
        ).as_dict()
        gas = node.require_fluid_properties().state_pt(
            gas_fluid, pressure, state["T_ull"]
        ).as_dict()
        liquid["V"] = state["m_liq"] / liquid["rho"]
        gas["V"] = state["m_ull"] / gas["rho"]
        liquid_energy = state["m_liq"] * liquid["u"]
        gas_energy = state["m_ull"] * gas["u"]
        fill = geometry.fill_state(liquid["V"])
        liquid_outlet_pressure = (
            pressure
            + liquid["rho"]
            * node.axial_specific_force
            * fill["fill_height"]
        )
        return {
            **state,
            "U_liq": liquid_energy,
            "U_ull": gas_energy,
            "P": pressure,
            "port_pressure": {
                "liquid": liquid_outlet_pressure,
                liquid_fluid: liquid_outlet_pressure,
                "ullage": pressure,
                "gas": pressure,
                gas_fluid: pressure,
            },
            "fluids": {
                liquid_fluid: {
                    **liquid,
                    "phase": "liquid",
                    "contact_area": fill["liquid_contact_area"],
                },
                gas_fluid: {
                    **gas,
                    "phase": "gas",
                    "contact_area": fill["ullage_contact_area"],
                },
            },
        }

    def output_state(self, node, evaluated) -> Dict[str, Any]:
        definition = node.definition
        geometry = definition["geometry"]
        liquid_volume = evaluated["fluids"][definition["liquid_fluid"]]["V"]
        return {
            **evaluated,
            "mass": evaluated["m_liq"] + evaluated["m_ull"],
            "tank_id": definition.get("tank_id"),
            "mode": node.mode,
            "axial_mass": geometry.axial_mass(
                liquid_volume=liquid_volume,
                liquid_mass=evaluated["m_liq"],
                ullage_mass=evaluated["m_ull"],
            ),
            **geometry.fill_state(liquid_volume),
        }

    def committed_state(self, node, state, evaluated) -> Dict[str, float]:
        return {
            name: float(evaluated[name])
            for name in (*self.names, "U_liq", "U_ull")
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
                (evaluated["m_liq"] - previous["m_liq"] - dt * mdot_liq)
                / liquid_mass_scale,
                (
                    evaluated["U_liq"]
                    - previous["U_liq"]
                    - dt * (hdot_liq + qdot_liq)
                    + pressure * dV_liq
                )
                / liquid_energy_scale,
                (evaluated["m_ull"] - previous["m_ull"] - dt * mdot_gas)
                / gas_mass_scale,
                (
                    evaluated["U_ull"]
                    - previous["U_ull"]
                    - dt * (hdot_gas + qdot_gas)
                    + pressure * dV_gas
                )
                / gas_energy_scale,
                (
                    evaluated["fluids"][liquid]["V"]
                    + evaluated["fluids"][gas]["V"]
                    - node.definition["geometry"].volume
                )
                / node.definition["geometry"].volume,
            ],
            dtype=float,
        )


class CombustionModel(SteadyModel):
    """Algebraic reacting chamber."""

    flow_coupled = True

    def trial_state(self, node, state, adjacent, node_states) -> Dict[str, Any]:
        definition = node.definition

        def inflow(fluid: str) -> float:
            return net_mdot(adjacent, fluid=fluid)

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
        prior = (
            node.evaluated.fluids.get(combustion_fluid)
            if node.evaluated is not None
            else None
        )
        transport = product if flowing else (
            prior.as_dict() if prior is not None else {"R": 300.0, "gamma": 1.2, "T": 300.0}
        )
        enthalpy = (
            transport["gamma"] * transport["R"] * transport["T"]
            / (transport["gamma"] - 1.0)
        )
        return {
            **state,
            "cstar": product["cstar"],
            "Cf": product["Cf"],
            "MR": mixture_ratio,
            "mdot_oxidizer": mdot_ox,
            "mdot_fuel": mdot_fuel,
            "fluids": {
                combustion_fluid: {
                    **{name: transport[name] for name in ("R", "gamma", "T")},
                    "h": enthalpy,
                    "phase": "gas",
                }
            },
        }

    def output_state(self, node, evaluated) -> Dict[str, Any]:
        return {
            **evaluated,
            "mode": node.mode,
            "shutdown_reason": node.shutdown_reason,
        }

    def residual(
        self, node, trial, previous, evaluated, adjacent, dt, heat_flux
    ) -> np.ndarray:
        return np.array(
            [
                net_mdot(adjacent)
                / self.continuity_scale(adjacent)
            ],
            dtype=float,
        )


class TwinPathJunctionModel(SteadyModel):
    """Shutdown chamber conserving gas and liquid on separate paths."""

    flow_coupled = True

    def __init__(self, phases):
        self.phases = tuple(phases)

    def state(self, node, dt, prescribed) -> Dict[str, float]:
        return super().state(node, dt, prescribed) if self.phases else {}

    def trial_state(self, node, state, adjacent, node_states) -> Dict[str, Any]:
        if not self.phases:
            state = {
                **state,
                "P": node_states[node.definition["ambient_node"]]["P"],
            }
        sources = {
            flow["fluid"].fluid: flow["fluid"]
            for incidence, flow in iter_flows(adjacent)
            if incidence * int(flow["direction"]) > 0
            and flow["fluid"].phase in self.phases
        }

        fluids = {}
        for fluid, source in sources.items():
            fluids[fluid] = FluidState.from_dict(
                fluid,
                {**source, "mdot": net_mdot(adjacent, fluid=fluid)},
            )
        return {
            **state,
            "cstar": 0.0,
            "Cf": 0.0,
            "MR": 0.0,
            "fluids": fluids,
        }

    def output_state(self, node, evaluated) -> Dict[str, Any]:
        return {
            **evaluated,
            "mode": node.mode,
            "shutdown_reason": node.shutdown_reason,
        }

    def residual(
        self, node, trial, previous, evaluated, adjacent, dt, heat_flux
    ) -> np.ndarray:
        return np.array(
            [
                net_mdot(adjacent, phase=phase)
                / self.continuity_scale(adjacent)
                for phase in self.phases
            ],
            dtype=float,
        )

    def outlet(self, node, evaluated, adjacent, port=None) -> Dict[str, FluidState]:
        fluids = evaluated.get("fluids", {})
        selected = node.fluid_for_port(port)
        if selected in fluids:
            fluids = {selected: fluids[selected]}
        if not fluids and self.phases:
            raise ValueError(f"Twin-path junction '{node.id}' has no inlet streams")
        return {
            name: fluid if isinstance(fluid, FluidState) else FluidState.from_dict(name, fluid)
            for name, fluid in fluids.items()
        }


class FluidNode:
    """Solver node holding state, connectivity, and an active physics model."""

    def __init__(self, node_id: str, definition: Dict[str, Any], model: NodeModel):
        self.id = node_id
        self.definition = definition
        self.model = model
        self.incoming: List[str] = []
        self.outgoing: List[str] = []
        self.state = NodeState.from_dict(self.model.initial_state(self))
        self.evaluated: Optional[NodeState] = None
        self.axial_specific_force = 0.0
        self.fluid_properties: Optional[PureFluidPropertySource] = None
        self.combustion_properties: Optional[CombustionPropertySource] = None

    @property
    def is_dynamic(self) -> bool:
        return self.model.is_dynamic

    def switch_model(self, model: NodeModel, state: Dict[str, Any]) -> None:
        self.model = model
        self.state = state if isinstance(state, NodeState) else NodeState.from_dict(state)

    def solver_state(self, dt, prescribed) -> Dict[str, float]:
        return self.model.state(self, dt, prescribed)

    def scales(self, state) -> Dict[str, float]:
        return self.model.scales(self, state)

    def trial_state(self, state, adjacent, node_states) -> Dict[str, Any]:
        return NodeState.from_dict(self.model.trial_state(self, state, adjacent, node_states))

    def output_state(self, evaluated) -> Dict[str, Any]:
        return NodeState.from_dict(self.model.output_state(self, evaluated))

    def event_values(self, evaluated) -> Dict[str, float]:
        return {}

    def apply_event(self, name: str) -> bool:
        raise ValueError(f"Node '{self.id}' has no event {name!r}")

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
        committed = (
            self.model.committed_state(self, state, evaluated)
            if evaluated is not None
            else dict(state)
        )
        self.state = committed if isinstance(committed, NodeState) else NodeState.from_dict(committed)
        if "quality" in self.state:
            quality = float(self.state["quality"])
            if not -1.0e-6 <= quality <= 1.01:
                raise ValueError(
                    f"Node '{self.id}' converged to invalid vapor quality {quality}"
                )
            self.state["quality"] = min(max(quality, 0.0), 1.0)
        if evaluated is not None:
            self.evaluated = evaluated if isinstance(evaluated, NodeState) else NodeState.from_dict(evaluated)

    def fluid_for_port(self, port: Optional[str]) -> Optional[str]:
        if port == "liquid":
            return self.definition.get("liquid_fluid", self.definition.get("fluid"))
        if port in ("ullage", "gas"):
            return self.definition.get("gas_fluid", self.definition.get("fluid"))
        if port == "oxidizer":
            return self.definition.get("oxidizer_fluid")
        if port == "fuel":
            return self.definition.get("fuel_fluid")
        return port

    def outlet(self, evaluated, adjacent, port=None):
        return self.model.outlet(self, evaluated, adjacent, port)

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


class VolumeComponent(FluidNode):
    """Component wrapper owning pure-fluid volume regime transitions."""

    def __init__(self, node_id: str, definition: Dict[str, Any], model: NodeModel):
        self._condensation_armed = True
        super().__init__(node_id, definition, model)

    def event_values(self, evaluated) -> Dict[str, float]:
        if not isinstance(self.model, VolumeModel):
            return {}
        if "quality" in self.state:
            return {"evaporate": 1.0 - float(evaluated["quality"])}
        fluid = self.definition["fluid"]
        if self.fluid_phase(fluid) != "gas":
            return {}
        properties = self.require_fluid_properties()
        if not properties.supports_saturation(fluid):
            return {}
        pressure = float(evaluated["P"])
        lower, upper = properties.saturation_bounds(fluid)
        if not lower <= pressure <= upper:
            return {}
        saturation = properties.saturation_at_p(fluid, pressure)
        if not self._condensation_armed:
            if float(evaluated["T"]) > saturation.T + 1.0e-6:
                self._condensation_armed = True
            else:
                return {}
        return {"condense": float(evaluated["T"]) - saturation.T}

    def apply_event(self, name: str) -> bool:
        if name == "condense":
            return self.condense()
        if name == "evaporate":
            return self.evaporate()
        return super().apply_event(name)

    def condense(self) -> bool:
        return isinstance(self.model, VolumeModel) and self.model.condense(self)

    def evaporate(self) -> bool:
        if "quality" not in self.state or self.state["quality"] < 1.0 - 1.0e-5:
            return False
        fluid = self.definition["fluid"]
        lower, upper = self.require_fluid_properties().saturation_bounds(fluid)
        saturation = self.require_fluid_properties().saturation_at_p(
            fluid, min(max(self.state["P"], lower), upper)
        )
        self.state = NodeState.from_dict(
            {
                "P": self.state["P"],
                "T": saturation.T,
                "m": self.state["m"],
                "U": self.state["U"],
            }
        )
        self._condensation_armed = False
        return True


class PressurantTankComponent(VolumeComponent):
    def __init__(self, node_id: str, definition: Dict[str, Any]):
        super().__init__(node_id, definition, VolumeModel())

    def fluid_phase(self, fluid: str) -> str:
        return "gas"


class PropellantTankComponent(VolumeComponent):
    dry_fraction = 1.0e-4
    max_dry_mass = 1.0e-4

    def __init__(self, node_id: str, definition: Dict[str, Any]):
        self.mode = "two_phase"
        initial_liquid_mass = float(definition["state0"]["m_liq"])
        self.dry_mass = max(
            min(initial_liquid_mass * self.dry_fraction, self.max_dry_mass),
            1.0e-12,
        )
        super().__init__(node_id, definition, TwoSpeciesModel())
        if initial_liquid_mass <= 0.0:
            self.dry_out()

    def event_values(self, evaluated) -> Dict[str, float]:
        if self.mode == "two_phase":
            return {"dryout": float(evaluated["m_liq"]) - self.dry_mass}
        return super().event_values(evaluated)

    def apply_event(self, name: str) -> bool:
        if name != "dryout":
            return super().apply_event(name)
        return self.dry_out()

    def dry_out(self) -> bool:
        if self.mode != "two_phase":
            return False
        if self.state["m_liq"] > self.dry_mass:
            return False
        self.mode = "gas"
        self.definition["fluid"] = self.definition["gas_fluid"]
        self.switch_model(
            VolumeModel(),
            {
                "P": self.state["P"],
                "T": self.state["T_ull"],
                "m": self.state["m_ull"],
                "U": self.state["U_ull"],
            },
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

    def update_mode(self, inflows: Adjacent) -> bool:
        phases = tuple(
            sorted(
                {
                    flow["fluid"].phase
                    for incidence, flow in iter_flows(inflows)
                    if incidence * int(flow["direction"]) > 0
                }
            )
        )
        if self.mode == "shutdown":
            if getattr(self.model, "phases", ()) == phases:
                return False
            self.switch_model(TwinPathJunctionModel(phases), self.state)
            return True
        fluid_names = {flow["fluid"].fluid for _, flow in iter_flows(inflows)}
        flow = {fluid: net_mdot(inflows, fluid=fluid) for fluid in fluid_names}
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
