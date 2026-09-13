from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import root

from .FluidBranch import (
    BangBangValveComponent,
    CompressibleLossModel,
    FluidBranch,
    IncompressibleLossModel,
    LossComponent,
    NozzleComponent,
    PumpComponent,
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
    VolumeComponent,
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
    """Assemble and solve residuals from component-selected model."""

    # move to advanced section of config
    residual_tolerance = 1.0e-7
    event_time_tolerance = 1.0e-5
    flow_direction_tolerance = 1.0e-10

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
    def _type(definition: Dict[str, Any]) -> str:
        fields = [name for name in ("component", "model") if name in definition]
        if len(fields) != 1:
            raise ValueError(
                "A network definition requires exactly one of component or model"
            )
        return str(definition[fields[0]])

    @classmethod
    def _make_node(cls, node_id: str, definition: Dict[str, Any]) -> FluidNode:
        type = cls._type(definition)
        if type == "boundary":
            return FluidNode(node_id, definition, BoundaryModel())
        if type == "junction":
            return FluidNode(node_id, definition, JunctionModel())
        if type == "pressurant_tank":
            return PressurantTankComponent(node_id, definition)
        if type == "propellant_tank":
            return PropellantTankComponent(node_id, definition)
        if type == "combustor":
            return CombustorComponent(node_id, definition)
        if type == "volume":
            return VolumeComponent(node_id, definition, VolumeModel())
        raise ValueError(f"Unsupported node component/model '{type}' for '{node_id}'")

    @classmethod
    def _make_branch(cls, branch_id: str, definition: Dict[str, Any]) -> FluidBranch:
        type = cls._type(definition)
        if type == "incompressible_loss":
            return FluidBranch(branch_id, definition, IncompressibleLossModel())
        if type == "compressible_loss":
            return FluidBranch(branch_id, definition, CompressibleLossModel())
        if type == "loss":
            return LossComponent(branch_id, definition)
        if type == "bang_bang_valve":
            return BangBangValveComponent(branch_id, definition)
        if type == "pump":
            return PumpComponent(branch_id, definition)
        if type == "nozzle":
            return NozzleComponent(branch_id, definition)
        raise ValueError(
            f"Unsupported branch component/model '{type}' for '{branch_id}'"
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

        bcs = bcs or {}
        heat_flux = heat_flux or {}
        x0: List[float] = []
        x_scale: List[float] = []
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
            scales = node.variable_scales(variables)
            x_scale.extend(scales[name] for name in names)

        for branch_id, branch in self.branch_objects.items():
            variables = branch.state_variables()
            names = tuple(variables)
            branch_layout[branch_id] = (
                names,
                slice(len(x0), len(x0) + len(names)),
            )
            x0.extend(variables.values())
            scales = branch.variable_scales(variables)
            x_scale.extend(scales[name] for name in names)

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

        def evaluate_candidate(x: np.ndarray):
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

            return (
                np.asarray(equations, dtype=float),
                raw_nodes,
                node_state,
                branch_state,
            )

        def residual(x: np.ndarray) -> np.ndarray:
            return evaluate_candidate(x)[0]

        if x0:
            scales = np.asarray(x_scale, dtype=float)

            def scaled_residual(values: np.ndarray) -> np.ndarray:
                return residual(values * scales)

            solution = root(
                scaled_residual,
                np.asarray(x0, dtype=float) / scales,
                method="hybr",
            )
            solved_values = solution.x * scales
            (
                final_residual,
                raw_nodes,
                node_state,
                branch_state,
            ) = evaluate_candidate(solved_values)
            residual_norm = float(np.max(np.abs(final_residual)))
            if (
                not np.isfinite(residual_norm)
                or residual_norm > self.residual_tolerance
            ):
                raise RuntimeError(
                    "Fluid network update failed: "
                    f"{solution.message}; scaled residual={residual_norm:.3e}"
                )
            message = (
                solution.message
                if solution.success
                else "Converged by scaled residual"
            )
        else:
            solved_values = np.empty(0)
            (
                final_residual,
                raw_nodes,
                node_state,
                branch_state,
            ) = evaluate_candidate(solved_values)
            message = "No active unknowns"

        td_state = {
            node_id: node.model.committed_state(
                node,
                raw_nodes[node_id],
                node_state[node_id],
            )
            for node_id, node in self.nodes.items()
            if node.is_dynamic
        }
        output_nodes = {
            node_id: node.output_state(node_state[node_id])
            for node_id, node in self.nodes.items()
        }
        candidate = NetworkState(
            node=output_nodes,
            br=branch_state,
            td=td_state,
        )
        if commit:
            self._commit_candidate(candidate, bcs)

        return {
            "success": True,
            "message": message,
            "node": candidate.node,
            "branch": candidate.br,
            "td_state": candidate.td,
            "mdot": {
                branch_id: self.branch_objects[branch_id].total_mdot(state)
                for branch_id, state in candidate.br.items()
            },
        }

    def _commit_candidate(
        self,
        candidate: NetworkState,
        bcs: Dict[str, Dict[str, Any]],
    ) -> None:
        """Commit an already-converged candidate without solving it again."""

        for node_id, node in self.nodes.items():
            if node_id not in bcs:
                state = (
                    candidate.td[node_id]
                    if node.is_dynamic
                    else candidate.node[node_id]
                )
                node.commit(state, candidate.node[node_id])
            else:
                node.evaluated = dict(candidate.node[node_id])
        for branch_id, branch in self.branch_objects.items():
            branch.commit(candidate.br[branch_id])
        self.state = candidate

    @staticmethod
    def _candidate(result: Dict[str, Any]) -> NetworkState:
        return NetworkState(
            node=result["node"],
            br=result["branch"],
            td=result["td_state"],
        )

    def _event_values(self, state: NetworkState) -> Dict[tuple, float]:
        values = {}
        for node_id, node in self.nodes.items():
            values.update(
                {
                    ("node", node_id, name): value
                    for name, value in node.event_values(
                        state.node[node_id]
                    ).items()
                }
            )
        for branch_id, branch in self.branch_objects.items():
            values.update(
                {
                    ("branch", branch_id, name): value
                    for name, value in branch.event_values(state.node).items()
                }
            )
        return values

    @staticmethod
    def _crossed(start: Dict[tuple, float], end: Dict[tuple, float]) -> List[tuple]:
        return [
            event
            for event, start_value in start.items()
            if start_value > 0.0 and end.get(event, start_value) <= 0.0
        ]

    def _event_step(self, dt, bcs, heat_flux):
        """Return the earliest event-aligned step and its converged candidate."""

        start_values = self._event_values(self.state)
        upper = float(dt)
        while True:
            try:
                upper_result = self._solve(upper, bcs, heat_flux, commit=False)
                break
            except (ValueError, RuntimeError):
                upper *= 0.5
                if upper <= self.event_time_tolerance:
                    raise
        upper_values = self._event_values(self._candidate(upper_result))
        crossed = self._crossed(start_values, upper_values)
        if not crossed:
            return upper, upper_result, None

        event = min(
            crossed,
            key=lambda key: start_values[key]
            / (start_values[key] - upper_values[key]),
        )
        fraction = start_values[event] / (
            start_values[event] - upper_values[event]
        )
        step = max(upper * fraction, self.event_time_tolerance)
        result = self._solve(step, bcs, heat_flux, commit=False)
        return step, result, event

    def _apply_event(self, event: tuple) -> bool:
        type, component_id, name = event
        component = (
            self.nodes[component_id]
            if type == "node"
            else self.branch_objects[component_id]
        )
        changed = component.apply_event(name)
        if changed and type == "node":
            evaluated = component.trial_state(component.state, [], {})
            component.evaluated = component.output_state(evaluated)
            self.state.node[component_id] = component.evaluated
            self.state.td[component_id] = dict(component.state)
        return changed

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
        if not self.state.node:
            self._solve(None, bcs, heat_flux, commit=True)
        while self._apply_transitions():
            self._solve(None, bcs, heat_flux, commit=True)

        remaining = dt
        result = None
        while remaining > self.event_time_tolerance:
            step, result, event = self._event_step(remaining, bcs, heat_flux)
            self._commit_candidate(self._candidate(result), bcs)
            remaining -= step

            if event is not None:
                self._apply_event(event)
                changed = self._apply_transitions()
                if event[0] == "node" or changed:
                    result = self._solve(None, bcs, heat_flux, commit=True)
            while self._apply_transitions():
                result = self._solve(None, bcs, heat_flux, commit=True)
            if event is None and remaining <= self.event_time_tolerance:
                break

        if result is None:
            result = self._solve(None, bcs, heat_flux, commit=True)
        return result

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
            if mdot >= -self.flow_direction_tolerance
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
            mdot = candidate.total_mdot(branch_variables[candidate_id])
            if candidate.enabled and mdot >= -self.flow_direction_tolerance:
                upstream.append((candidate_id, candidate.definition["from"], 1.0))
        for candidate_id in node.outgoing:
            if candidate_id == excluded_branch:
                continue
            candidate = self.branch_objects[candidate_id]
            mdot = candidate.total_mdot(branch_variables[candidate_id])
            if candidate.enabled and mdot < -self.flow_direction_tolerance:
                upstream.append((candidate_id, candidate.definition["to"], -1.0))
        if not upstream:
            raise ValueError(f"Node '{node_id}' has no incoming fluid stream")

        connections = []
        for candidate_id, source_id, sign in upstream:
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
            connections.append(FlowConn(candidate_id, sign, transported))
        components = node.model.outlet_state(
            node, node_state[node_id], connections, preferred_fluid
        )
        return FlowConn(
            excluded_branch,
            1.0,
            {"components": components, "enabled": True},
        )

    def _apply_transitions(self) -> bool:
        changed = False
        due = [
            event
            for event, value in self._event_values(self.state).items()
            if value <= 0.0
        ]
        if due:
            for event in due:
                changed = self._apply_event(event) or changed

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
