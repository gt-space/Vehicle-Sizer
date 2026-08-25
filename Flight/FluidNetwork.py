from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import root

from .FluidsDef import FluidsDef


@dataclass
class NetworkState:
    node: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    br: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    td: Dict[str, Dict[str, float]] = field(default_factory=dict)


class FluidNode:
    """Base interface implemented by every node type."""

    def __init__(self, node_id: str, definition: Dict[str, Any]) -> None:
        self.id = node_id
        self.definition = definition
        self.incoming: List[str] = []
        self.outgoing: List[str] = []
        self.state: Dict[str, float] = {}

    @property
    def is_dynamic(self) -> bool:
        return False

    def initial_state(self) -> Dict[str, float]:
        return {}

    def initialize(self) -> None:
        self.state = self.initial_state()

    def state_variables(
        self,
        dt: Optional[float],
        prescribed: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        return {}

    def trial_state(
        self,
        variables: Dict[str, float],
        prescribed: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return dict(prescribed or self.state)

    def evaluate(self, state: Dict[str, float]) -> Dict[str, Any]:
        return dict(state)

    def coupled_evaluate(
        self,
        state: Dict[str, Any],
        node_state: Dict[str, Dict[str, Any]],
        branch_state: Dict[str, Dict[str, float]],
        branches: Dict[str, "FluidBranch"],
    ) -> Dict[str, Any]:
        return state

    def residual(
        self,
        trial: Dict[str, float],
        previous: Dict[str, float],
        evaluated: Dict[str, Any],
        adjacent: List[Dict[str, Any]],
        dt: Optional[float],
        heat_flux: Dict[str, float],
    ) -> np.ndarray:
        return np.empty(0)

    def commit(self, state: Dict[str, float]) -> None:
        self.state = dict(state)


class AlgebraicNode(FluidNode):
    """Zero-volume junction that contributes an algebraic mass balance."""

    def initial_state(self) -> Dict[str, float]:
        return {"P": float(self.definition.get("P0", 0.0))}

    def state_variables(self, dt, prescribed) -> Dict[str, float]:
        return {} if prescribed is not None else {"P": float(self.state["P"])}

    def trial_state(self, variables, prescribed) -> Dict[str, Any]:
        return dict(prescribed if prescribed is not None else variables)

    def residual(self, trial, previous, evaluated, adjacent, dt, heat_flux) -> np.ndarray:
        mdot_net = sum(item["sign"] * item["state"]["mdot"] for item in adjacent)
        return np.array([mdot_net], dtype=float)


class CombustionNode(AlgebraicNode):
    """Algebraic chamber with CEA combustion properties."""

    def coupled_evaluate(
        self,
        state: Dict[str, Any],
        node_state: Dict[str, Dict[str, Any]],
        branch_state: Dict[str, Dict[str, float]],
        branches: Dict[str, "FluidBranch"],
    ) -> Dict[str, Any]:
        def inflow(fluid: str) -> float:
            total = 0.0
            for bid in self.incoming:
                if branches[bid].definition.get("fluid") == fluid:
                    total += branch_state[bid]["mdot"]
            for bid in self.outgoing:
                if branches[bid].definition.get("fluid") == fluid:
                    total -= branch_state[bid]["mdot"]
            return total

        design = dict(self.definition["design_state"])
        mdot_oxidizer = inflow(self.definition.get("oxidizer_fluid", "ox"))
        mdot_fuel = inflow(self.definition.get("fuel_fluid", "fuel"))
        mixture_ratio = (
            mdot_oxidizer / mdot_fuel
        )
        chamber_pressure = float(state["P"])
        ambient_pressure = node_state[
            self.definition.get("ambient_node", "ambient")
        ]["P"]
        if chamber_pressure <= 0.0:
            performance = design
        else:
            performance = FluidsDef.combustion_properties(
                chamber_pressure=chamber_pressure,
                mixture_ratio=mixture_ratio,
                ambient_pressure=ambient_pressure,
                expansion_ratio=self.definition["expansion_ratio"],
                cea=self.definition["cea"],
                cstar_efficiency=self.definition.get("cstar_efficiency", 1.0),
                cf_efficiency=self.definition.get("cf_efficiency", 1.0),
            )

        fluid_state = {
            name: performance[name]
            for name in ("R", "gamma", "T", "h")
            if name in performance
        }
        return {
            **state,
            "cstar": performance["cstar"],
            "Cf": performance["Cf"],
            "MR": mixture_ratio,
            "mdot_oxidizer": mdot_oxidizer,
            "mdot_fuel": mdot_fuel,
            "fluids": {
                self.definition.get("combustion_fluid", "combustion_gas"): fluid_state
            },
        }


class BoundaryNode(FluidNode):
    """Prescribed node; it adds no unknown and no residual equation."""

    def trial_state(self, variables, prescribed) -> Dict[str, Any]:
        state = prescribed if prescribed is not None else self.definition
        if "P" not in state:
            raise ValueError(f"Boundary node '{self.id}' requires a pressure state")
        return dict(state)

class DynamicNode(FluidNode):
    """Base class for nodes whose state propagates in time."""

    @property
    def is_dynamic(self) -> bool:
        return True

    def state_variables(self, dt, prescribed) -> Dict[str, float]:
        if dt is None or prescribed is not None:
            return {}
        return dict(self.state)

    def trial_state(self, variables, prescribed) -> Dict[str, Any]:
        if prescribed is not None:
            return dict(prescribed)
        return dict(variables or self.state)

    @staticmethod
    def _fluxes(
        adjacent: List[Dict[str, Any]],
        fluid: Optional[str] = None,
    ) -> Tuple[float, float]:
        mdot = 0.0
        hdot = 0.0
        for item in adjacent:
            if fluid is not None and item["branch"].get("fluid") != fluid:
                continue
            signed_mdot = item["sign"] * item["state"]["mdot"]
            mdot += signed_mdot
            hdot += signed_mdot * item["state"]["h"]
        return mdot, hdot

    def _heat_flux(self, heat_flux: Dict[str, float]) -> float:
        """Return the surface heat flux applied to this node [W/m^2]."""
        return float(heat_flux.get(self.id, 0.0))


class GasVolumeNode(DynamicNode):
    """Dynamic gas volume such as a COPV."""

    state_names = ("m", "U")

    def initial_state(self) -> Dict[str, float]:
        state0 = self.definition["state0"]
        return {"m": float(state0["m"]), "U": float(state0["U"])}

    def evaluate(self, state: Dict[str, float]) -> Dict[str, Any]:
        geometry = self.definition["geometry"]
        density = state["m"] / geometry.volume
        internal_energy = state["U"] / state["m"]
        fluid_state = FluidsDef.coolprop_state(
            self.definition["fluid"],
            "Dmass",
            density,
            "Umass",
            internal_energy,
        )
        fluid_state["V"] = geometry.volume
        return {
            **state,
            "P": fluid_state["P"],
            "fluids": {self.definition["fluid"]: fluid_state},
        }

    def residual(self, trial, previous, evaluated, adjacent, dt, heat_flux) -> np.ndarray:
        if dt is None:
            return np.empty(0)
        mdot, hdot = self._fluxes(adjacent)
        q_flux = self._heat_flux(heat_flux)
        qdot = q_flux * float(self.definition["geometry"].internal_area)
        return np.array(
            [
                trial["m"] - previous["m"] - dt * mdot,
                trial["U"] - previous["U"] - dt * (hdot + qdot),
            ],
            dtype=float,
        )

class PropellantTankNode(DynamicNode):
    """Dynamic liquid-propellant tank with a gas ullage."""

    state_names = ("m_liq", "U_liq", "m_ull", "U_ull")

    def initial_state(self) -> Dict[str, float]:
        state0 = self.definition["state0"]
        return {
            name: float(state0[name])
            for name in ("m_liq", "U_liq", "m_ull", "U_ull")
        }

    def evaluate(self, state: Dict[str, float]) -> Dict[str, Any]:
        geometry = self.definition["geometry"]
        liquid_fluid = self.definition["liquid_fluid"]
        gas_fluid = self.definition["gas_fluid"]
        tank = FluidsDef.tank_compatibility(
            m_liquid=state["m_liq"],
            U_liquid=state["U_liq"],
            m_gas=state["m_ull"],
            U_gas=state["U_ull"],
            tank_volume=geometry.volume,
            liquid_fluid=liquid_fluid,
            gas_fluid=gas_fluid,
            pressure_guess=self.definition["P0"],
        )
        fill = geometry.fill_state(tank["liquid"]["V"])

        return {
            **state,
            "P": tank["P"],
            **fill,
            "fluids": {
                liquid_fluid: {
                    **tank["liquid"],
                    "contact_area": fill["liquid_contact_area"],
                },
                gas_fluid: {
                    **tank["gas"],
                    "contact_area": fill["ullage_contact_area"],
                },
            },
        }

    def residual(self, trial, previous, evaluated, adjacent, dt, heat_flux) -> np.ndarray:
        if dt is None:
            return np.empty(0)
        liquid_fluid = self.definition["liquid_fluid"]
        gas_fluid = self.definition["gas_fluid"]
        mdot_liq, hdot_liq = self._fluxes(adjacent, liquid_fluid)
        mdot_ull, hdot_ull = self._fluxes(adjacent, gas_fluid)
        q_flux = self._heat_flux(heat_flux)
        qdot_liq = q_flux * evaluated["fluids"][liquid_fluid]["contact_area"]
        qdot_ull = q_flux * evaluated["fluids"][gas_fluid]["contact_area"]
        return np.array(
            [
                trial["m_liq"] - previous["m_liq"] - dt * mdot_liq,
                trial["U_liq"] - previous["U_liq"] - dt * (hdot_liq + qdot_liq),
                trial["m_ull"] - previous["m_ull"] - dt * mdot_ull,
                trial["U_ull"] - previous["U_ull"] - dt * (hdot_ull + qdot_ull),
            ],
            dtype=float,
        )

class FluidBranch:
    """Base interface for a branch with mass flow as its algebraic unknown."""

    def __init__(self, branch_id: str, definition: Dict[str, Any]) -> None:
        self.id = branch_id
        self.definition = definition
        self.state = {"mdot": float(definition.get("design_mdot", 0.0))}

    def state_variables(self) -> Dict[str, float]:
        return {"mdot": self.state["mdot"]}

    def evaluate(
        self,
        variables: Dict[str, float],
        node_state: Dict[str, Dict[str, Any]],
        get_property,
    ) -> Dict[str, Any]:
        mdot = float(variables["mdot"])
        donor = self.definition["from"] if mdot >= 0.0 else self.definition["to"]
        return {
            "mdot": mdot,
            "h": get_property(donor, self.definition["fluid"], "h"),
            "dP": (
                node_state[self.definition["from"]]["P"]
                - node_state[self.definition["to"]]["P"]
            ),
        }

    def residual(
        self,
        state: Dict[str, Any],
        node_state: Dict[str, Dict[str, Any]],
        get_property,
    ) -> np.ndarray:
        raise NotImplementedError

    def commit(self, state: Dict[str, Any]) -> None:
        self.state = {"mdot": float(state["mdot"])}

    def flow_scale(self) -> float:
        return max(abs(float(self.definition.get("design_mdot", 0.0))), 1.0)


class LiquidBranch(FluidBranch):
    """Incompressible restriction."""

    def residual(self, state, node_state, get_property) -> np.ndarray:
        donor = self.definition["from"] if state["mdot"] >= 0.0 else self.definition["to"]
        density = get_property(donor, self.definition["fluid"], "rho")
        expected = FluidsDef.incompressible_mdot(
            self.definition["CdA"], density, state["dP"]
        )
        state["rho"] = density
        return np.array([(state["mdot"] - expected) / self.flow_scale()])


class GasBranch(FluidBranch):
    """Compressible restriction with automatic choking."""

    def effective_cda(self) -> float:
        return float(self.definition["CdA"])

    def residual(self, state, node_state, get_property) -> np.ndarray:
        direction = 1.0 if state["dP"] >= 0.0 else -1.0
        donor = self.definition["from"] if direction > 0.0 else self.definition["to"]
        P_upstream = max(
            node_state[self.definition["from"]]["P"],
            node_state[self.definition["to"]]["P"],
        )
        P_downstream = min(
            node_state[self.definition["from"]]["P"],
            node_state[self.definition["to"]]["P"],
        )
        expected = direction * FluidsDef.compressible_mdot(
            self.effective_cda(),
            P_upstream,
            P_downstream,
            get_property(donor, self.definition["fluid"], "T"),
            get_property(donor, self.definition["fluid"], "R"),
            get_property(donor, self.definition["fluid"], "gamma"),
        )
        return np.array([(state["mdot"] - expected) / self.flow_scale()])


class BangBangBranch(GasBranch):
    """Duty-cycled gas orifice used to pressurize a propellant tank."""

    def effective_cda(self) -> float:
        return float(self.definition["CdA"] * self.definition["duty_cycle"])


class PumpBranch(FluidBranch):
    """Pressure-rise branch with an optional internal flow resistance."""

    def residual(self, state, node_state, get_property) -> np.ndarray:
        mdot = state["mdot"]
        head_model = self.definition.get("head_model")
        head = head_model(mdot) if callable(head_model) else self.definition["dP"]
        loss = 0.0
        if self.definition.get("CdA") is not None:
            donor = self.definition["from"] if mdot >= 0.0 else self.definition["to"]
            density = get_property(donor, self.definition["fluid"], "rho")
            loss = np.sign(mdot) * (mdot / self.definition["CdA"]) ** 2 / (2.0 * density)
        pressure_residual = state["dP"] + head - loss
        scale = max(abs(float(head)), 1.0e5)
        return np.array([pressure_residual / scale])


class NozzleBranch(FluidBranch):
    """Rocket nozzle using the current chamber characteristic velocity."""

    def residual(self, state, node_state, get_property) -> np.ndarray:
        chamber = node_state[self.definition["from"]]
        expected = 0.0
        if state["dP"] > 0.0:
            expected = (
                self.definition.get("Cd", 1.0)
                * chamber["P"]
                * self.definition["At"]
                / chamber["cstar"]
            )
        return np.array([(state["mdot"] - expected) / self.flow_scale()])


class FluidNetwork:
    """Assemble and solve residuals supplied by node and branch subclasses."""

    def __init__(
        self,
        nodes: Dict[str, Dict[str, Any]],
        branches: Dict[str, Dict[str, Any]],
    ) -> None:
        self.node_definitions = nodes
        self.branches = branches
        self.nodes = {
            nid: self._make_node(nid, definition) for nid, definition in nodes.items()
        }
        self.branch_objects = {
            bid: self._make_branch(bid, definition)
            for bid, definition in branches.items()
        }
        for node in self.nodes.values():
            node.initialize()
        self.state = NetworkState()
        self._connect()

    @staticmethod
    def _make_node(node_id: str, definition: Dict[str, Any]) -> FluidNode:
        node_class = definition.get("node_class")
        if node_class is not None:
            return node_class(node_id, definition)
        node_type = definition.get("type")
        if node_type == "boundary_pressure":
            return BoundaryNode(node_id, definition)
        if node_type == "comb_device":
            return CombustionNode(node_id, definition)
        if node_type == "gas_volume":
            return GasVolumeNode(node_id, definition)
        if node_type == "propellant_tank":
            return PropellantTankNode(node_id, definition)
        return AlgebraicNode(node_id, definition)

    @staticmethod
    def _make_branch(branch_id: str, definition: Dict[str, Any]) -> FluidBranch:
        branch_class = definition.get("branch_class")
        if branch_class is not None:
            return branch_class(branch_id, definition)
        branch_type = definition["type"]
        if branch_type in ("liquid_loss", "liquid_orifice"):
            return LiquidBranch(branch_id, definition)
        if branch_type == "gas_orifice":
            return GasBranch(branch_id, definition)
        if branch_type == "bang_bang":
            return BangBangBranch(branch_id, definition)
        if branch_type == "pump":
            return PumpBranch(branch_id, definition)
        if branch_type == "nozzle":
            return NozzleBranch(branch_id, definition)
        raise ValueError(f"Unsupported branch type '{branch_type}' for '{branch_id}'")

    def _connect(self) -> None:
        for bid, branch in self.branches.items():
            if branch["from"] not in self.nodes or branch["to"] not in self.nodes:
                raise ValueError(f"Branch '{bid}' references an unknown node")
            self.nodes[branch["from"]].outgoing.append(bid)
            self.nodes[branch["to"]].incoming.append(bid)

    def update(
        self,
        dt: Optional[float] = None,
        bcs: Optional[Dict[str, Dict[str, Any]]] = None,
        heat_flux: Optional[Dict[str, float]] = None,
        commit: bool = True,
    ) -> Dict[str, Any]:
        """Solve all active node states and all branch mass flows."""

        bcs = bcs or {}
        heat_flux = heat_flux or {}
        x0: List[float] = []
        node_layout = {}
        branch_layout = {}

        for nid, node in self.nodes.items():
            variables = node.state_variables(dt, bcs.get(nid))
            names = tuple(variables)
            node_layout[nid] = (names, slice(len(x0), len(x0) + len(names)))
            x0.extend(variables.values())

        for bid, branch in self.branch_objects.items():
            variables = branch.state_variables()
            names = tuple(variables)
            branch_layout[bid] = (names, slice(len(x0), len(x0) + len(names)))
            x0.extend(variables.values())

        def unpack(x: np.ndarray):
            raw_nodes = {}
            for nid, node in self.nodes.items():
                names, indices = node_layout[nid]
                values = {name: float(value) for name, value in zip(names, x[indices])}
                raw_nodes[nid] = node.trial_state(values, bcs.get(nid))
            raw_branches = {}
            for bid in self.branch_objects:
                names, indices = branch_layout[bid]
                raw_branches[bid] = {
                    name: float(value) for name, value in zip(names, x[indices])
                }
            return raw_nodes, raw_branches

        def residual(x: np.ndarray) -> np.ndarray:
            raw_nodes, raw_branches = unpack(x)
            node_state = {
                nid: node.evaluate(raw_nodes[nid])
                for nid, node in self.nodes.items()
            }
            for nid, node in self.nodes.items():
                node_state[nid] = node.coupled_evaluate(
                    node_state[nid],
                    node_state,
                    raw_branches,
                    self.branch_objects,
                )

            get_property = lambda nid, fluid, name: self._property(
                nid, fluid, name, node_state, set()
            )
            branch_state = {
                bid: branch.evaluate(raw_branches[bid], node_state, get_property)
                for bid, branch in self.branch_objects.items()
            }

            equations: List[float] = []
            for nid, node in self.nodes.items():
                names, _ = node_layout[nid]
                if not names:
                    continue
                adjacent = [
                    {
                        "branch": self.branches[bid],
                        "state": branch_state[bid],
                        "sign": sign,
                    }
                    for sign, branch_ids in ((1.0, node.incoming), (-1.0, node.outgoing))
                    for bid in branch_ids
                ]
                equations.extend(
                    node.residual(
                        raw_nodes[nid],
                        node.state,
                        node_state[nid],
                        adjacent,
                        dt,
                        heat_flux,
                    )
                )
            for bid, branch in self.branch_objects.items():
                equations.extend(
                    branch.residual(branch_state[bid], node_state, get_property)
                )

            td_state = {
                nid: raw_nodes[nid]
                for nid, node in self.nodes.items()
                if node.is_dynamic
            }
            self.state = NetworkState(node=node_state, br=branch_state, td=td_state)
            return np.asarray(equations, dtype=float)

        if x0:
            solution = root(residual, np.asarray(x0, dtype=float), method="hybr")
            if not solution.success:
                raise RuntimeError(f"Fluid network update failed: {solution.message}")
            residual(solution.x)
            solved_values = solution.x
            message = solution.message
        else:
            solved_values = np.empty(0)
            residual(solved_values)
            message = "No active unknowns"

        if commit:
            raw_nodes, _ = unpack(solved_values)
            for nid, node in self.nodes.items():
                if node.is_dynamic:
                    if dt is not None and nid not in bcs:
                        node.commit(raw_nodes[nid])
                elif nid not in bcs:
                    node.commit(self.state.node[nid])
            for bid, branch in self.branch_objects.items():
                branch.commit(self.state.br[bid])

        return {
            "success": True,
            "message": message,
            "node": self.state.node,
            "branch": self.state.br,
            "td_state": self.state.td,
            "mdot": {bid: state["mdot"] for bid, state in self.state.br.items()},
        }

    def _property(
        self,
        node_id: str,
        fluid: str,
        name: str,
        node_state: Dict[str, Dict[str, Any]],
        visited: set,
    ) -> float:
        if node_id in visited:
            raise KeyError(f"No upstream {name!r} property found for fluid {fluid!r}")
        visited.add(node_id)
        fluid_state = node_state[node_id].get("fluids", {}).get(fluid, {})
        if name in fluid_state:
            return float(fluid_state[name])
        for bid in self.nodes[node_id].incoming:
            branch = self.branches[bid]
            if branch.get("fluid") == fluid:
                try:
                    return self._property(
                        branch["from"], fluid, name, node_state, visited
                    )
                except KeyError:
                    continue
        raise KeyError(
            f"Node '{node_id}' has no upstream {name!r} property for fluid {fluid!r}"
        )
