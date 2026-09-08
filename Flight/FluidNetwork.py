from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import root

from .FluidBranch import (
    CompressibleLossModel,
    FluidBranch,
    IncompressibleLossModel,
    LossComponent,
    NozzleComponent,
    PumpComponent,
    ValveComponent,
)
from .FluidNode import (
    BoundaryModel,
    CombustorComponent,
    FlowConn,
    FluidNode,
    JunctionModel,
    PressurantTankComponent,
    PropellantTankComponent,
    SteadyModel,
    VolumeModel,
)
from FluidProperties.PropertyModels import (
    CombustionPropertySource,
    PureFluidPropertySource,
)

@dataclass
class NetworkState:
    node: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    br: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    td: Dict[str, Dict[str, float]] = field(default_factory=dict)


class FluidNetwork:
    """Assemble and solve residuals from component-selected physics models."""

    def __init__(
        self,
        nodes: Dict[str, Dict[str, Any]],
        branches: Dict[str, Dict[str, Any]],
        fluid_properties: Optional[PureFluidPropertySource] = None,
        combustion_properties: Optional[CombustionPropertySource] = None,
    ) -> None:
        self.node_definitions = nodes
        self.branches = branches
        self.nodes = {
            node_id: self._make_node(node_id, definition)
            for node_id, definition in nodes.items()
        }
        for node in self.nodes.values():
            node.fluid_properties = fluid_properties
            node.combustion_properties = combustion_properties
        self.branch_objects = {
            branch_id: self._make_branch(branch_id, definition)
            for branch_id, definition in branches.items()
        }
        self.state = NetworkState()
        self._connect()

    @staticmethod
    def _kind(definition: Dict[str, Any]) -> str:
        fields = [name for name in ("component", "model") if name in definition]
        if len(fields) != 1:
            raise ValueError(
                "A network definition requires exactly one of component or model"
            )
        return str(definition[fields[0]])

    @classmethod
    def _make_node(cls, node_id: str, definition: Dict[str, Any]) -> FluidNode:
        kind = cls._kind(definition)
        if kind == "boundary":
            return FluidNode(node_id, definition, BoundaryModel())
        if kind == "junction":
            return FluidNode(node_id, definition, JunctionModel())
        if kind == "pressurant_tank":
            return PressurantTankComponent(node_id, definition)
        if kind == "propellant_tank":
            return PropellantTankComponent(node_id, definition)
        if kind == "combustor":
            return CombustorComponent(node_id, definition)
        if kind == "volume":
            return FluidNode(node_id, definition, VolumeModel())
        raise ValueError(f"Unsupported node component/model '{kind}' for '{node_id}'")

    @classmethod
    def _make_branch(cls, branch_id: str, definition: Dict[str, Any]) -> FluidBranch:
        kind = cls._kind(definition)
        if kind == "incompressible_loss":
            return FluidBranch(branch_id, definition, IncompressibleLossModel())
        if kind == "compressible_loss":
            return FluidBranch(branch_id, definition, CompressibleLossModel())
        if kind == "loss":
            return LossComponent(branch_id, definition)
        if kind == "bang_bang_valve":
            return ValveComponent(branch_id, definition)
        if kind == "pump":
            return PumpComponent(branch_id, definition)
        if kind == "nozzle":
            return NozzleComponent(branch_id, definition)
        raise ValueError(
            f"Unsupported branch component/model '{kind}' for '{branch_id}'"
        )

    def _connect(self) -> None:
        for branch_id, branch in self.branches.items():
            if branch["from"] not in self.nodes or branch["to"] not in self.nodes:
                raise ValueError(f"Branch '{branch_id}' references an unknown node")
            self.nodes[branch["from"]].outgoing.append(branch_id)
            self.nodes[branch["to"]].incoming.append(branch_id)

    def _solve(
        self,
        dt: Optional[float] = None,
        bcs: Optional[Dict[str, Dict[str, Any]]] = None,
        heat_flux: Optional[Dict[str, float]] = None,
        commit: bool = True,
    ) -> Dict[str, Any]:
        """Solve all active node states and branch mass flows."""

        bcs = bcs or {}
        heat_flux = heat_flux or {}
        x0: List[float] = []
        node_layout = {}
        branch_layout = {}

        for node_id, node in self.nodes.items():
            connected = any(
                self.branch_objects[branch_id].enabled
                for branch_id in (*node.incoming, *node.outgoing)
            )
            prescribed = bcs.get(node_id)
            if isinstance(node.model, SteadyModel) and not connected:
                prescribed = node.state
            variables = node.state_variables(dt, prescribed)
            names = tuple(variables)
            node_layout[node_id] = (names, slice(len(x0), len(x0) + len(names)))
            x0.extend(variables.values())

        for branch_id, branch in self.branch_objects.items():
            variables = branch.state_variables()
            names = tuple(variables)
            branch_layout[branch_id] = (
                names,
                slice(len(x0), len(x0) + len(names)),
            )
            x0.extend(variables.values())

        def unpack(x: np.ndarray):
            raw_nodes = {}
            for node_id, node in self.nodes.items():
                names, indices = node_layout[node_id]
                values = {
                    name: float(value) for name, value in zip(names, x[indices])
                }
                raw_nodes[node_id] = {
                    **node.state,
                    **values,
                    **bcs.get(node_id, {}),
                }
            raw_branches = {}
            for branch_id in self.branch_objects:
                names, indices = branch_layout[branch_id]
                raw_branches[branch_id] = {
                    name: float(value) for name, value in zip(names, x[indices])
                }
            return raw_nodes, raw_branches

        def adjacent(node, branch_states):
            return [
                FlowConn(branch_id, sign, branch_states[branch_id])
                for sign, branch_ids in (
                    (1.0, node.incoming),
                    (-1.0, node.outgoing),
                )
                for branch_id in branch_ids
            ]

        def residual(x: np.ndarray) -> np.ndarray:
            raw_nodes, raw_branches = unpack(x)
            trial_branches = {
                branch_id: self.branch_objects[branch_id].flow_state(state)
                for branch_id, state in raw_branches.items()
            }
            node_state = {
                node_id: node.trial_state(
                    raw_nodes[node_id],
                    adjacent(node, trial_branches),
                    raw_nodes,
                )
                for node_id, node in self.nodes.items()
            }
            def evaluate_branches(include_coupled=True):
                states = dict(trial_branches)
                for branch_id, branch in self.branch_objects.items():
                    source = self.nodes[branch.definition["from"]]
                    if not include_coupled and source.model.flow_coupled:
                        continue
                    inlet = None
                    if branch.tracks_stream:
                        inlet = self._inlet_connection(
                            branch_id,
                            raw_branches[branch_id],
                            raw_branches,
                            node_state,
                        )
                    states[branch_id] = branch.evaluate(
                        raw_branches[branch_id], node_state, inlet
                    )
                return states

            branch_state = evaluate_branches(include_coupled=False)
            for node_id, node in self.nodes.items():
                if node.model.flow_coupled:
                    node_state[node_id] = node.trial_state(
                        raw_nodes[node_id],
                        adjacent(node, branch_state),
                        node_state,
                    )
            branch_state = evaluate_branches()

            equations: List[float] = []
            for node_id, node in self.nodes.items():
                names, _ = node_layout[node_id]
                if names:
                    equations.extend(
                        node.residual(
                            raw_nodes[node_id],
                            node.state,
                            node_state[node_id],
                            adjacent(node, branch_state),
                            dt,
                            heat_flux,
                        )
                    )
            for branch_id, branch in self.branch_objects.items():
                equations.extend(
                    branch.residual(branch_state[branch_id], node_state)
                )

            self.state = NetworkState(
                node=node_state,
                br=branch_state,
                td={
                    node_id: raw_nodes[node_id]
                    for node_id, node in self.nodes.items()
                    if node.is_dynamic
                },
            )
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

        invalid = [
            node_id
            for node_id, state in self.state.node.items()
            if state.get("_trial_valid") is False
        ]
        if invalid:
            states = {node_id: self.state.td[node_id] for node_id in invalid}
            raise RuntimeError(
                f"Fluid network converged to invalid node states: {states}"
            )
        for state in self.state.node.values():
            state.pop("_trial_valid", None)

        if commit:
            raw_nodes, _ = unpack(solved_values)
            for node_id, node in self.nodes.items():
                if node.is_dynamic:
                    if dt is not None and node_id not in bcs:
                        node.commit(raw_nodes[node_id], self.state.node[node_id])
                elif node_id not in bcs:
                    node.commit(self.state.node[node_id], self.state.node[node_id])
            for branch_id, branch in self.branch_objects.items():
                branch.commit(self.state.br[branch_id])
        elif dt is None:
            for node_id, node in self.nodes.items():
                node.evaluated = dict(self.state.node[node_id])

        return {
            "success": True,
            "message": message,
            "node": self.state.node,
            "branch": self.state.br,
            "td_state": self.state.td,
            "mdot": {
                branch_id: self.branch_objects[branch_id].total_mdot(state)
                for branch_id, state in self.state.br.items()
            },
        }

    def update(
        self,
        dt: Optional[float] = None,
        bcs: Optional[Dict[str, Dict[str, Any]]] = None,
        heat_flux: Optional[Dict[str, float]] = None,
        commit: bool = True,
    ) -> Dict[str, Any]:
        """Propagate the network and apply component mode transitions."""

        if dt is None or not commit:
            return self._solve(dt, bcs, heat_flux, commit)
        if dt < 0.0:
            raise ValueError("Fluid-network dt cannot be negative")

        bcs = bcs or {}
        heat_flux = heat_flux or {}
        if not self.state.br:
            self._solve(None, bcs, heat_flux, commit=False)
        for branch_id, branch in self.branch_objects.items():
            branch.commit(self.state.br[branch_id])
        while self._apply_transitions():
            self._solve(None, bcs, heat_flux, commit=True)

        remaining = dt
        result = None
        while remaining > 0.0:
            event = self._next_dryout(remaining)
            step = event if event is not None else remaining
            if step > 0.0:
                result = self._solve(step, bcs, heat_flux, commit=True)
                remaining -= step

            while self._apply_transitions():
                result = self._solve(None, bcs, heat_flux, commit=True)
            if event is None:
                break

        if result is None:
            result = self._solve(None, bcs, heat_flux, commit=True)
        return result

    def _connections(
        self,
        node: FluidNode,
        branch_states: Dict[str, Dict[str, Any]],
    ) -> List[FlowConn]:
        return [
            FlowConn(
                branch_id,
                sign,
                {
                    **branch_states[branch_id],
                    "enabled": self.branch_objects[branch_id].enabled,
                },
            )
            for sign, branch_ids in ((1.0, node.incoming), (-1.0, node.outgoing))
            for branch_id in branch_ids
        ]

    def _inlet_connection(
        self,
        branch_id: str,
        variables: Dict[str, float],
        branch_variables: Dict[str, Dict[str, float]],
        node_state: Dict[str, Dict[str, Any]],
    ) -> FlowConn:
        branch = self.branch_objects[branch_id]
        mdot = branch.total_mdot(variables)
        donor = (
            branch.definition["from"]
            if mdot >= 0.0
            else branch.definition["to"]
        )
        return self._outlet_connection(
            donor,
            branch.fluid,
            branch_id,
            branch_variables,
            node_state,
            set(),
        )

    def _outlet_connection(
        self,
        node_id: str,
        preferred_fluid: str,
        excluded_branch: str,
        branch_variables: Dict[str, Dict[str, float]],
        node_state: Dict[str, Dict[str, Any]],
        visited: set,
    ) -> FlowConn:
        if node_id in visited:
            raise ValueError(f"Cannot resolve a fluid stream through node '{node_id}'")
        visited = {*visited, node_id}
        node = self.nodes[node_id]
        if node_state[node_id].get("fluids") or not node.model.routes_inlet:
            components = node.model.outlet_state(
                node, node_state[node_id], [], preferred_fluid
            )
            return FlowConn(
                excluded_branch,
                1.0,
                {"components": components, "enabled": True},
            )

        upstream = []
        for candidate_id in node.incoming:
            if candidate_id == excluded_branch:
                continue
            candidate = self.branch_objects[candidate_id]
            if candidate.enabled and candidate.total_mdot(branch_variables[candidate_id]) >= 0.0:
                upstream.append((candidate_id, candidate.definition["from"]))
        for candidate_id in node.outgoing:
            if candidate_id == excluded_branch:
                continue
            candidate = self.branch_objects[candidate_id]
            if candidate.enabled and candidate.total_mdot(branch_variables[candidate_id]) < 0.0:
                upstream.append((candidate_id, candidate.definition["to"]))
        if not upstream:
            raise ValueError(f"Node '{node_id}' has no incoming fluid stream")

        connections = []
        for candidate_id, source_id in upstream:
            source = self._outlet_connection(
                source_id,
                preferred_fluid,
                candidate_id,
                branch_variables,
                node_state,
                visited,
            )
            transported = self.branch_objects[candidate_id].flow_state(
                branch_variables[candidate_id], source
            )
            connections.append(FlowConn(candidate_id, 1.0, transported))
        components = node.model.outlet_state(
            node, node_state[node_id], connections, preferred_fluid
        )
        return FlowConn(
            excluded_branch,
            1.0,
            {"components": components, "enabled": True},
        )

    def _next_dryout(self, dt: float):
        events = []
        for node in self.nodes.values():
            if not isinstance(node, PropellantTankComponent):
                continue
            event_time = node.time_to_dry(self._connections(node, self.state.br))
            if event_time is not None and event_time <= dt:
                events.append(event_time)
        return min(events) if events else None

    def _apply_transitions(self) -> bool:
        changed = False
        for node in self.nodes.values():
            if not isinstance(node, PropellantTankComponent):
                continue
            if node.dry_out():
                node.evaluated = node.trial_state(node.state, [], {})
                self.state.node[node.id] = node.evaluated
                changed = True

        for node in self.nodes.values():
            if not isinstance(node, CombustorComponent):
                continue
            variables = {
                branch_id: branch.state
                for branch_id, branch in self.branch_objects.items()
            }
            inflows = []
            for branch_id in node.incoming:
                branch = self.branch_objects[branch_id]
                inlet = self._inlet_connection(
                    branch_id,
                    variables[branch_id],
                    variables,
                    self.state.node,
                )
                inflows.append(
                    FlowConn(
                        branch_id,
                        1.0,
                        branch.flow_state(variables[branch_id], inlet),
                    )
                )
            if node.update_mode(inflows):
                changed = True
            phases = getattr(node.model, "phases", ())
            for branch_id in node.outgoing:
                branch = self.branch_objects[branch_id]
                if isinstance(branch, NozzleComponent):
                    if branch.set_mode(node.mode == "combusting", phases):
                        changed = True
        return changed
