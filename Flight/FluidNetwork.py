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
    TwinPathNozzleModel,
)
from .FluidNode import (
    BoundaryModel,
    CombustorComponent,
    FluidNode,
    JunctionModel,
    PressurantTankComponent,
    PropellantTankComponent,
    SteadyModel,
    VolumeComponent,
    VolumeModel,
)
from .FluidState import BranchState, NodeState
from FluidProperties.PropertyModels import (
    CombustionPropertySource,
    PureFluidPropertySource,
)

@dataclass
class NetworkState:
    nodes: Dict[str, NodeState] = field(default_factory=dict)
    branches: Dict[str, BranchState] = field(default_factory=dict)

    @property
    def node(self):
        return self.nodes

    @property
    def br(self):
        return self.branches


class FluidNetwork:
    """Assemble and solve residuals from component-selected physics models."""

    residual_tolerance = 1.0e-7
    event_time_tolerance = 1.0e-5
    flow_direction_tolerance = 1.0e-10

    def __init__(
        self,
        nodes: Dict[str, Dict[str, Any]],
        branches: Dict[str, Dict[str, Any]],
        fluid_properties: Optional[PureFluidPropertySource] = None,
        combustion_properties: Optional[CombustionPropertySource] = None,
        tolerances: Optional[Dict[str, float]] = None,
    ) -> None:
        tolerances = tolerances or {
            "residual_tolerance": 1.0e-7,
            "event_time_tolerance": 1.0e-5,
            "flow_direction_tolerance": 1.0e-10,
        }
        self.residual_tolerance = float(tolerances["residual_tolerance"])
        self.event_time_tolerance = float(tolerances["event_time_tolerance"])
        self.flow_direction_tolerance = float(
            tolerances["flow_direction_tolerance"]
        )
        if min(
            self.residual_tolerance,
            self.event_time_tolerance,
            self.flow_direction_tolerance,
        ) <= 0.0:
            raise ValueError("Fluid-network tolerances must be positive")
        self.nodes = {
            node_id: self._make_node(node_id, definition)
            for node_id, definition in nodes.items()
        }
        for node in self.nodes.values():
            node.fluid_properties = fluid_properties
            node.combustion_properties = combustion_properties
        self.branches = {
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
            return VolumeComponent(node_id, definition, VolumeModel())
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
            return BangBangValveComponent(branch_id, definition)
        if kind == "pump":
            return PumpComponent(branch_id, definition)
        if kind == "nozzle":
            return NozzleComponent(branch_id, definition)
        raise ValueError(
            f"Unsupported branch component/model '{kind}' for '{branch_id}'"
        )

    def _connect(self) -> None:
        for branch_id, branch in self.branches.items():
            if branch.from_node not in self.nodes or branch.to_node not in self.nodes:
                raise ValueError(f"Branch '{branch_id}' references an unknown node")
            self.nodes[branch.from_node].outgoing.append(branch_id)
            self.nodes[branch.to_node].incoming.append(branch_id)

    def _adjacent(self, node: FluidNode, states: Dict[str, BranchState]):
        return [
            (incidence, states[branch_id])
            for incidence, branch_ids in ((1.0, node.incoming), (-1.0, node.outgoing))
            for branch_id in branch_ids
            if branch_id in states
        ]

    def _initial_directions(self) -> Dict[str, Dict[str, int]]:
        directions = {}
        for branch_id, branch in self.branches.items():
            if isinstance(branch.model, TwinPathNozzleModel):
                previous = next(iter(branch.state.flows.values()), {}).get("direction", 1)
                directions[branch_id] = {
                    phase: int(previous) for phase in branch.model.phases
                }
            elif branch.state.flows:
                directions[branch_id] = {
                    name: int(flow["direction"])
                    for name, flow in branch.state.flows.items()
                }
            else:
                directions[branch_id] = {"main": 1}
        return directions

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
        x_scale: List[float] = []
        node_layout = {}
        branch_layout = {}

        for node_id, node in self.nodes.items():
            connected = any(
                self.branches[branch_id].enabled
                for branch_id in (*node.incoming, *node.outgoing)
            )
            prescribed = bcs.get(node_id)
            if isinstance(node.model, SteadyModel) and not connected:
                prescribed = node.state
            variables = node.solver_state(dt, prescribed)
            names = tuple(variables)
            node_layout[node_id] = (names, slice(len(x0), len(x0) + len(names)))
            x0.extend(variables.values())
            scales = node.scales(variables)
            x_scale.extend(scales[name] for name in names)

        for branch_id, branch in self.branches.items():
            variables = branch.solver_state()
            names = tuple(variables)
            branch_layout[branch_id] = (
                names,
                slice(len(x0), len(x0) + len(names)),
            )
            x0.extend(variables.values())
            scales = branch.scales(variables)
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
            for branch_id in self.branches:
                names, indices = branch_layout[branch_id]
                raw_branches[branch_id] = {
                    name: float(value) for name, value in zip(names, x[indices])
                }
            return raw_nodes, raw_branches

        directions = self._initial_directions()

        def evaluate_candidate(x: np.ndarray):
            raw_nodes, raw_branches = unpack(x)
            node_state = {
                node_id: node.trial_state(
                    raw_nodes[node_id],
                    [],
                    raw_nodes,
                )
                for node_id, node in self.nodes.items()
            }
            branch_state: Dict[str, BranchState] = {}
            ready_nodes = {
                node_id
                for node_id, node in self.nodes.items()
                if not node.model.routes_inlet and not node.model.flow_coupled
            }
            pending = set(self.branches)
            while pending:
                progressed = False
                for node_id, node in self.nodes.items():
                    if node_id in ready_nodes:
                        continue
                    required = {
                        branch_id
                        for incidence, branch_ids in ((1, node.incoming), (-1, node.outgoing))
                        for branch_id in branch_ids
                        if any(incidence * direction > 0 for direction in directions[branch_id].values())
                        and self.branches[branch_id].enabled
                    }
                    if required <= branch_state.keys():
                        node_state[node_id] = node.trial_state(
                            raw_nodes[node_id], self._adjacent(node, branch_state), node_state
                        )
                        ready_nodes.add(node_id)
                        progressed = True

                for branch_id in tuple(pending):
                    branch = self.branches[branch_id]
                    donors = {
                        branch.from_node if direction > 0 else branch.to_node
                        for direction in directions[branch_id].values()
                    }
                    if not donors <= ready_nodes:
                        continue
                    source_fluids = {}
                    for direction in directions[branch_id].values():
                        donor = branch.from_node if direction > 0 else branch.to_node
                        port = branch.from_port if direction > 0 else branch.to_port
                        source_fluids.update(
                            self.nodes[donor].outlet(
                                node_state[donor],
                                self._adjacent(self.nodes[donor], branch_state),
                                port,
                            )
                        )
                    branch_state[branch_id] = branch.evaluate(
                        raw_branches[branch_id], node_state, source_fluids, directions[branch_id]
                    )
                    pending.remove(branch_id)
                    progressed = True
                if not progressed:
                    raise ValueError(
                        "Fluid transport contains an algebraic cycle without a state-owning node: "
                        + ", ".join(sorted(pending))
                    )

            equations: List[float] = []
            equation_counts = {}
            for node_id, node in self.nodes.items():
                names, _ = node_layout[node_id]
                if names:
                    values = node.residual(
                        raw_nodes[node_id],
                        node.state,
                        node_state[node_id],
                        self._adjacent(node, branch_state),
                        dt,
                        heat_flux,
                    )
                    equations.extend(values)
                    equation_counts[f"node:{node_id}"] = (len(names), len(values))
            for branch_id, branch in self.branches.items():
                values = branch.residual(branch_state[branch_id], node_state)
                equations.extend(values)
                equation_counts[f"branch:{branch_id}"] = (
                    len(branch_layout[branch_id][0]), len(values)
                )

            if len(equations) != len(x0):
                raise RuntimeError(
                    "Fluid network has unequal unknown and residual counts: "
                    + ", ".join(
                        f"{name}={unknowns}/{residuals}"
                        for name, (unknowns, residuals) in equation_counts.items()
                        if unknowns != residuals
                    )
                )

            return (
                np.asarray(equations, dtype=float),
                raw_nodes,
                node_state,
                branch_state,
            )

        def residual(x: np.ndarray) -> np.ndarray:
            return evaluate_candidate(x)[0]

        scales = np.asarray(x_scale, dtype=float)
        solved_values = np.asarray(x0, dtype=float)
        message = "No active unknowns"
        for direction_pass in range(len(self.branches) + 2):
            if x0:

                def scaled_residual(values: np.ndarray) -> np.ndarray:
                    return residual(values * scales)

                solution = root(
                    scaled_residual,
                    solved_values / scales,
                    method="hybr",
                    options={"xtol": min(self.residual_tolerance, 1.0e-10)},
                )
                solved_values = solution.x * scales
            (
                final_residual,
                raw_nodes,
                node_state,
                branch_state,
            ) = evaluate_candidate(solved_values)
            residual_norm = float(np.max(np.abs(final_residual))) if final_residual.size else 0.0
            if (
                not np.isfinite(residual_norm)
                or residual_norm > self.residual_tolerance
            ):
                raise RuntimeError(
                    "Fluid network update failed: "
                    f"{solution.message}; scaled residual={residual_norm:.3e}"
                )
            if x0:
                message = solution.message if solution.success else "Converged by scaled residual"
            changed = False
            for branch_id, state in branch_state.items():
                branch = self.branches[branch_id]
                if isinstance(branch.model, TwinPathNozzleModel):
                    solved = {
                        phase: state.phase_mdot(phase)
                        for phase in branch.model.phases
                    }
                else:
                    solved = {"main": state.mdot}
                for name, mdot in solved.items():
                    if abs(mdot) > self.flow_direction_tolerance:
                        direction = 1 if mdot > 0.0 else -1
                        if directions[branch_id][name] != direction:
                            directions[branch_id][name] = direction
                            changed = True
            if not changed:
                break
        else:
            raise RuntimeError("Fluid flow directions did not settle")

        output_nodes = {
            node_id: node.output_state(node_state[node_id])
            for node_id, node in self.nodes.items()
        }
        candidate = NetworkState(
            nodes=output_nodes,
            branches=branch_state,
        )
        if commit:
            self._commit_candidate(candidate, bcs)

        return {
            "success": True,
            "message": message,
            "state": candidate,
            "node": {name: state.as_dict() for name, state in candidate.nodes.items()},
            "branch": {name: state.as_dict() for name, state in candidate.branches.items()},
            "td_state": {name: state.state for name, state in candidate.nodes.items() if self.nodes[name].is_dynamic},
            "mdot": {
                branch_id: self.branches[branch_id].total_mdot(state)
                for branch_id, state in candidate.branches.items()
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
                node.commit(candidate.nodes[node_id], candidate.nodes[node_id])
            else:
                node.evaluated = candidate.nodes[node_id]
        for branch_id, branch in self.branches.items():
            branch.commit(candidate.branches[branch_id])
        self.state = candidate

    @staticmethod
    def _candidate(result: Dict[str, Any]) -> NetworkState:
        return result["state"]

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
        for branch_id, branch in self.branches.items():
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
        kind, component_id, name = event
        component = (
            self.nodes[component_id]
            if kind == "node"
            else self.branches[component_id]
        )
        changed = component.apply_event(name)
        if changed and kind == "node":
            evaluated = component.trial_state(component.state, [], {})
            component.evaluated = component.output_state(evaluated)
            self.state.nodes[component_id] = component.evaluated
        return changed

    def update(
        self,
        dt: Optional[float] = None,
        bcs: Optional[Dict[str, Dict[str, Any]]] = None,
        heat_flux: Optional[Dict[str, float]] = None,
        commit: bool = True,
        axial_specific_force: float = 0.0,
    ) -> Dict[str, Any]:
        """Propagate the network and apply component mode transitions."""

        axial_specific_force = float(axial_specific_force)
        if not np.isfinite(axial_specific_force):
            raise ValueError("Axial specific force must be finite")
        for node in self.nodes.values():
            node.axial_specific_force = axial_specific_force

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
            inflows = self._adjacent(node, self.state.branches)
            if node.update_mode(inflows):
                changed = True
            phases = getattr(node.model, "phases", ())
            for branch_id in node.outgoing:
                branch = self.branches[branch_id]
                if isinstance(branch, NozzleComponent):
                    if branch.set_mode(node.mode == "combusting", phases):
                        changed = True
        return changed
