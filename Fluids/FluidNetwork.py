from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from constraints import OperatingInfeasible, merge_margins
from .FluidBranch import (
    BangBangValveComponent,
    CompressibleLossModel,
    FluidBranch,
    IncompressibleLossModel,
    LossComponent,
    NozzleComponent,
    PumpComponent,
    TwinPathNozzleModel,
    CombustionNozzleModel,
    PumpModel,
    IdealRegulatorModel,
    RegulatorComponent,
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
    iter_flows,
    TwoSpeciesModel,
    CombustionModel,
    TwinPathJunctionModel,
)
from .FluidState import BranchState, NodeState
from FluidProperties.PropertyModels import (
    CombustionPropertySource,
    PureFluidPropertySource,
)
from FluidProperties.LookupTables import LookupBoundsError


@dataclass
class NetworkState:
    nodes: Dict[str, NodeState] = field(default_factory=dict)
    branches: Dict[str, BranchState] = field(default_factory=dict)

    @property
    def node(self):
        return self.nodes


class FluidNetwork:
    """Assemble and solve residuals from component-selected physics models."""

    def __init__(
        self,
        nodes: Dict[str, Dict[str, Any]],
        branches: Dict[str, Dict[str, Any]],
        fluid_properties: Optional[PureFluidPropertySource] = None,
        combustion_properties: Optional[CombustionPropertySource] = None,
        tolerances: Optional[Dict[str, float]] = None,
        constraint_monitor=None,
        stop_at_shutdown: bool = False,
        stop_at_triple_point: bool = False,
    ) -> None:
        if not isinstance(stop_at_shutdown, bool):
            raise ValueError("stop_at_shutdown must be boolean")
        self.stop_at_shutdown = stop_at_shutdown
        if not isinstance(stop_at_triple_point, bool):
            raise ValueError("stop_at_triple_point must be boolean")
        self.stop_at_triple_point = stop_at_triple_point
        self.frozen = False
        self._triple_points = {}
        tolerances = tolerances or {
            "residual_tolerance": 1.0e-7,
            "event_time_tolerance": 1.0e-5,
            "flow_direction_tolerance": 1.0e-10,
        }
        self.residual_tolerance = float(tolerances["residual_tolerance"])
        self.sparse_jacobian = tolerances.get("sparse_jacobian", False)
        if not isinstance(self.sparse_jacobian, bool):
            raise ValueError("sparse_jacobian must be boolean")
        self._jacobian_cache = None
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
        self.time = 0.0
        self.events = []
        self.event_counts = {}
        self.constraint_monitor = constraint_monitor
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
        if kind == "regulator":
            return RegulatorComponent(branch_id, definition)
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

    def _jacobian_sparsity(self, node_layout, branch_layout, counts, directions):
        """Conservative dependencies for built-in models, including transported state."""

        node_models = (BoundaryModel, JunctionModel, VolumeModel, TwoSpeciesModel,
                       CombustionModel, TwinPathJunctionModel)
        branch_models = (IncompressibleLossModel, CompressibleLossModel, PumpModel,
                         CombustionNozzleModel, TwinPathNozzleModel, IdealRegulatorModel)
        node_components = (FluidNode, VolumeComponent, PressurantTankComponent,
                           PropellantTankComponent, CombustorComponent)
        branch_components = (FluidBranch, LossComponent, PumpComponent,
                             NozzleComponent, BangBangValveComponent, RegulatorComponent)
        if any(type(n.model) not in node_models or type(n) not in node_components
               for n in self.nodes.values()) or any(
            type(b.model) not in branch_models or type(b) not in branch_components
            for b in self.branches.values()
        ):
            return None  # Unknown physics remains dense rather than assuming false zeros.
        key = (
            tuple((k, names, type(self.nodes[k].model),
                   getattr(self.nodes[k].model, "phases", ()),
                   self.nodes[k].definition.get("ambient_node"))
                  for k, (names, _) in node_layout.items()),
            tuple((k, names, type(self.branches[k].model),
                   self.branches[k].from_node, self.branches[k].to_node,
                   self.branches[k].from_port, self.branches[k].to_port,
                   tuple(directions[k].items())) for k, (names, _) in branch_layout.items()),
            tuple(counts.items()),
        )
        if self._jacobian_cache is not None and self._jacobian_cache[0] == key:
            return self._jacobian_cache[1]
        nc = {k: set(range(s.start, s.stop)) for k, (_, s) in node_layout.items()}
        bc = {k: set(range(s.start, s.stop)) for k, (_, s) in branch_layout.items()}
        donors = {
            k: {b.from_node if d > 0 else b.to_node for d in directions[k].values()}
            for k, b in self.branches.items()
        }
        fluid = {k: set(columns) for k, columns in nc.items()}
        for k, node in self.nodes.items():
            if isinstance(node.model, (JunctionModel, TwinPathJunctionModel)):
                fluid[k] = set()
            if isinstance(node.model, (CombustionModel, TwinPathJunctionModel)):
                fluid[k].update(nc.get(node.definition.get("ambient_node"), ()))
                for branch_id in (*node.incoming, *node.outgoing):
                    fluid[k].update(bc[branch_id])
        # Forward routed thermodynamic dependencies, not downstream pressure feedback.
        changed = True
        while changed:
            changed = False
            for k, node in self.nodes.items():
                if not isinstance(node.model, (JunctionModel, TwinPathJunctionModel)):
                    continue
                previous = len(fluid[k])
                for branch_id in (*node.incoming, *node.outgoing):
                    if donors[branch_id] - {k}:
                        fluid[k].update(bc[branch_id])
                        for donor in donors[branch_id] - {k}:
                            fluid[k].update(fluid[donor])
                changed |= len(fluid[k]) != previous

        def flow_columns(branch_id):
            columns = set(bc[branch_id])
            for donor in donors[branch_id]:
                columns.update(fluid[donor])
            return columns

        def pressure_columns(node_id, port):
            names, indices = node_layout[node_id]
            if port is not None:
                return nc[node_id]  # Includes liquid-level/acceleration port pressure.
            if "P" in names:
                return {indices.start + names.index("P")}
            return nc.get(self.nodes[node_id].definition.get("ambient_node"), set())

        rows, columns = [], []
        offset = 0
        for label, (_, size) in counts.items():
            kind, component_id = label.split(":", 1)
            if kind == "node":
                node = self.nodes[component_id]
                dependencies = set(nc[component_id]) if node.model.is_dynamic else set()
                for branch_id in (*node.incoming, *node.outgoing):
                    dependencies.update(flow_columns(branch_id) if node.model.is_dynamic else bc[branch_id])
            else:
                branch = self.branches[component_id]
                dependencies = flow_columns(component_id)
                dependencies.update(pressure_columns(branch.from_node, branch.from_port))
                dependencies.update(pressure_columns(branch.to_node, branch.to_port))
            for row in range(offset, offset + size):
                rows.extend([row] * len(dependencies))
                columns.extend(dependencies)
            offset += size
        unknowns = sum(len(c) for c in (*nc.values(), *bc.values()))
        pattern = csr_matrix((np.ones(len(rows), dtype=bool), (rows, columns)),
                             shape=(offset, unknowns))
        self._jacobian_cache = (key, pattern)
        return pattern

    def _solve(
        self,
        dt: Optional[float] = None,
        bcs: Optional[Dict[str, Dict[str, Any]]] = None,
        heat_rate: Optional[Dict[str, Any]] = None,
        commit: bool = True,
        detect_modes: bool = False,
    ) -> Dict[str, Any]:
        """Solve all active node states and branch mass flows."""

        bcs = bcs or {}
        heat_rate = heat_rate or {}
        x0: List[float] = []
        x_scale: List[float] = []
        lower_bounds: List[float] = []
        upper_bounds: List[float] = []
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
            bounds = node.bounds(variables)
            lower_bounds.extend(bounds[name][0] for name in names)
            upper_bounds.extend(bounds[name][1] for name in names)
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
            bounds = branch.bounds(variables)
            lower_bounds.extend(bounds[name][0] for name in names)
            upper_bounds.extend(bounds[name][1] for name in names)
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
        equation_labels = []
        equation_counts = {}

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
            for branch_id, branch in self.branches.items():
                if branch.enabled:
                    continue
                branch_state[branch_id] = deepcopy(branch.state)
                branch_state[branch_id].enabled = False
                for flow in branch_state[branch_id].flows.values():
                    flow["mdot"] = 0.0
            ready_nodes = {
                node_id
                for node_id, node in self.nodes.items()
                if not node.model.routes_inlet and not node.model.flow_coupled
            }
            pending = {
                branch_id
                for branch_id, branch in self.branches.items()
                if branch.enabled
            }
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
            equation_labels.clear()
            equation_counts.clear()
            for node_id, node in self.nodes.items():
                names, _ = node_layout[node_id]
                if names:
                    values = node.residual(
                        raw_nodes[node_id],
                        node.state,
                        node_state[node_id],
                        self._adjacent(node, branch_state),
                        dt,
                        heat_rate,
                    )
                    equations.extend(values)
                    equation_labels.extend(
                        f"node:{node_id}[{index}]" for index in range(len(values))
                    )
                    equation_counts[f"node:{node_id}"] = (len(names), len(values))
            for branch_id, branch in self.branches.items():
                values = branch.residual(branch_state[branch_id], node_state)
                equations.extend(values)
                equation_labels.extend(
                    f"branch:{branch_id}[{index}]" for index in range(len(values))
                )
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

        if detect_modes:
            _, _, initial_nodes, initial_branches = evaluate_candidate(np.asarray(x0, dtype=float))
            changed = self._apply_mode_transitions(initial_branches, initial_nodes)
            if self.stop_at_shutdown and self._is_shutdown():
                return self._freeze_shutdown(initial_nodes, initial_branches, bcs, commit)
            if changed:
                return self._solve(dt, bcs, heat_rate, commit)

        scales = np.asarray(x_scale, dtype=float)
        lower = np.asarray(lower_bounds, dtype=float) / scales
        upper = np.asarray(upper_bounds, dtype=float) / scales
        initial_values = np.asarray(x0, dtype=float)
        # Liquid loss/pump equations permit signed pressure trials without a
        # property lookup at that pressure. Check the converged inlet pressure
        # before commit; bounding it at zero can trap an impossible design in
        # thousands of least-squares iterations instead of identifying it.
        evaluate_candidate(initial_values)
        liquid_inlets = {branch.from_node for branch in self.branches.values()
                         if branch.enabled and isinstance(branch.model, PumpModel)
                         and isinstance(self.nodes[branch.from_node].model, JunctionModel)}
        for node_id in liquid_inlets:
            lower[node_layout[node_id][1]] = -np.inf
        solved_values = initial_values.copy()
        solve_message = "no algebraic unknowns"
        for _ in range(len(self.branches) + 2):
            if x0:

                def scaled_residual(values: np.ndarray) -> np.ndarray:
                    return residual(values * scales)

                sparsity = None
                # Scalar problems gain nothing from sparse differentiation; SciPy's
                # sparse trust-region path also requires a two-dimensional subspace.
                if self.sparse_jacobian and len(solved_values) > 1:
                    evaluate_candidate(solved_values)
                    sparsity = self._jacobian_sparsity(
                        node_layout, branch_layout, equation_counts, directions
                    )
                solution = least_squares(
                    scaled_residual,
                    solved_values / scales,
                    method="trf",
                    bounds=(lower, upper),
                    xtol=min(self.residual_tolerance, 1.0e-12),
                    ftol=min(self.residual_tolerance, 1.0e-12),
                    gtol=min(self.residual_tolerance, 1.0e-12),
                    # Try the alternate method promptly when TRF makes slow
                    # progress near a regulator's pressure-reversal boundary.
                    max_nfev=100,
                    jac_sparsity=sparsity,
                    tr_options=(
                        {"atol": min(self.residual_tolerance, 1.0e-12),
                         "btol": min(self.residual_tolerance, 1.0e-12),
                         "maxiter": 10 * len(solved_values)}
                        if sparsity is not None else {}
                    ),
                )
                solved_values = solution.x * scales
                solve_message = solution.message
            (
                final_residual,
                raw_nodes,
                node_state,
                branch_state,
            ) = evaluate_candidate(solved_values)
            residual_norm = float(np.max(np.abs(final_residual))) if final_residual.size else 0.0
            for retry in range(3) if x0 else ():
                if np.isfinite(residual_norm) and residual_norm <= self.residual_tolerance:
                    break
                if retry == 0:
                    seed = initial_values
                elif retry == 1:
                    seed = self._transition_seed(
                        solved_values, node_layout, node_state, branch_layout, branch_state
                    )
                else:
                    seed = solved_values
                if seed is None:
                    continue
                fallback = least_squares(
                    scaled_residual,
                    seed / scales,
                    method="trf" if retry == 2 else "dogbox",
                    bounds=(lower, upper),
                    xtol=min(self.residual_tolerance, 1.0e-12),
                    ftol=min(self.residual_tolerance, 1.0e-12),
                    gtol=min(self.residual_tolerance, 1.0e-12),
                    max_nfev=5000,
                )
                fallback_values = fallback.x * scales
                fallback_candidate = evaluate_candidate(fallback_values)
                fallback_norm = float(np.max(np.abs(fallback_candidate[0])))
                if np.isfinite(fallback_norm) and (
                    not np.isfinite(residual_norm)
                    or fallback_norm < residual_norm
                ):
                    solved_values = fallback_values
                    (final_residual, raw_nodes, node_state, branch_state) = fallback_candidate
                    residual_norm = fallback_norm
                    solve_message = fallback.message
            if (
                not np.isfinite(residual_norm)
                or residual_norm > self.residual_tolerance
            ):
                worst = int(np.argmax(np.abs(final_residual)))
                limits = []
                for node_id, (names, indices) in node_layout.items():
                    node = self.nodes[node_id]
                    if isinstance(node.model, VolumeModel) and "quality" in names:
                        pressure = float(node_state[node_id]["P"])
                        low, high = node.bounds(raw_nodes[node_id])["P"]
                        if np.isclose(pressure, low, rtol=1e-8, atol=0.0) or np.isclose(
                            pressure, high, rtol=1e-8, atol=0.0
                        ):
                            limits.append(
                                f"{node_id} saturation pressure {pressure:.6g} Pa "
                                f"at property-domain limit [{low:.6g}, {high:.6g}] Pa"
                            )
                raise RuntimeError(
                    "Fluid network update failed: "
                    f"{solve_message}; scaled residual={residual_norm:.3e} "
                    f"at {equation_labels[worst]}"
                    + ("; " + "; ".join(limits) if limits else "")
                )
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

        invalid_pressures = {f"node.{key}.Pmin": float(node_state[key]["P"])
                             for key in liquid_inlets if node_state[key]["P"] < 0.0}
        if invalid_pressures:
            raise OperatingInfeasible(invalid_pressures, self.time + (dt or 0.0))

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

        return self._output(candidate)

    def _output(self, candidate):
        return {
            "state": candidate,
            "node": {name: state.as_dict() for name, state in candidate.nodes.items()},
            "branch": {name: state.as_dict() for name, state in candidate.branches.items()},
            "td_state": {
                name: state.state
                for name, state in candidate.nodes.items()
                if self.nodes[name].model.is_dynamic
            },
            "mdot": {
                branch_id: self.branches[branch_id].total_mdot(state)
                for branch_id, state in candidate.branches.items()
            },
        }

    def _is_shutdown(self):
        chambers = [node for node in self.nodes.values() if isinstance(node, CombustorComponent)]
        return bool(chambers) and all(node.mode == "shutdown" for node in chambers)

    def _freeze_shutdown(self, node_states, branch_states, bcs, commit):
        """Close the feed paths at cutoff without solving post-shutdown hydraulics."""
        candidate = NetworkState(
            nodes={key: node.output_state({**node_states[key], **node.state, **bcs.get(key, {})})
                   for key, node in self.nodes.items()},
            branches=deepcopy(branch_states),
        )
        for state in candidate.branches.values():
            state.enabled = False
            for flow in state.flows.values():
                flow["mdot"] = 0.0
            for name in state.state:
                if name.startswith("mdot_"):
                    state.state[name] = 0.0
            state.metadata.update(thrust=0.0, pump_active=False)
        if commit:
            for branch in self.branches.values():
                branch.enabled = False
            self._commit_candidate(candidate, bcs)
        return self._output(candidate)

    def _transition_seed(self, values, node_layout, node_states, branch_layout, branch_states):
        """Escape transition plateaus without changing equations or bounds.

        A passive gas junction cannot sustain pressure outside all its neighbors.
        After a pump loses its liquid supply, stale pressure guesses can land on
        just such a plateau. Move only those guesses to the neighboring midpoint;
        the subsequent solve must still meet the original residual tolerance.
        """
        seed = values.copy()
        changed = False
        # An ideal regulator's interior residual depends only on pressure. At a
        # zero-time transition, conserved tank mass/energy can fix pressure just
        # off target, leaving no flow derivative to reach the required limit.
        for branch_id, branch in self.branches.items():
            if not isinstance(branch, RegulatorComponent) or not branch.enabled:
                continue
            target = float(branch.parameters["target_pressure"])
            error = (target - node_states[branch.to_node]["P"]) / target
            if abs(error) <= 16 * np.finfo(float).eps:
                continue
            _, indices = branch_layout[branch_id]
            flow = branch_states[branch_id]["max_mdot"] if error > 0 else 0.0
            if np.any(seed[indices] != flow):
                seed[indices] = flow
                changed = True
        for node_id, node in self.nodes.items():
            names, indices = node_layout[node_id]
            if not isinstance(node.model, JunctionModel) or names != ("P",):
                continue
            branches = [self.branches[key] for key in (*node.incoming, *node.outgoing)
                        if self.branches[key].enabled]
            if not branches or any(type(branch.model) is not CompressibleLossModel for branch in branches):
                continue
            pressures = [float(node_states[branch.from_node if branch.to_node == node_id
                                          else branch.to_node]["P"]) for branch in branches]
            low, high = min(pressures), max(pressures)
            pressure = float(values[indices][0])
            if low <= pressure <= high:
                continue
            seed[indices] = 0.5 * low + 0.5 * high
            changed = True
        return seed if changed else None

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

    def _event_values(self, state: NetworkState) -> Dict[tuple, float]:
        if self.frozen:
            return {}
        values = {}
        for node_id, node in self.nodes.items():
            if self.stop_at_triple_point and node.model.is_dynamic:
                evaluated = state.node[node_id]
                for fluid in evaluated.fluids:
                    properties = node.require_fluid_properties()
                    if not properties.supports_saturation(fluid):
                        continue
                    if fluid not in self._triple_points:
                        pressure = properties.saturation_bounds(fluid)[0]
                        self._triple_points[fluid] = (
                            pressure, properties.saturation_at_p(fluid, pressure).T
                        )
                    pressure, temperature = self._triple_points[fluid]
                    if "quality" in node.state:
                        margin = float(evaluated["P"]) / pressure - 1.0
                    else:
                        field = ("T_ull" if fluid == node.definition.get("gas_fluid")
                                 else "T_liq") if "T_ull" in evaluated else "T"
                        margin = float(evaluated[field]) / temperature - 1.0
                    values[("node", node_id, f"triple_point:{fluid}")] = margin
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

    def _predict_event_values(self, dt, bcs, heat_rate):
        """Explicit Euler proposal from local balance residuals; never commit it."""

        predicted = deepcopy(self.state)
        snapshot = self._snapshot()
        try:
            for node_id, node in self.nodes.items():
                if not node.model.is_dynamic or node_id in bcs:
                    continue
                variables = node.solver_state(0.0, None)
                if not variables:
                    continue
                adjacent = self._adjacent(node, self.state.branches)
                names = tuple(variables)
                scales = node.scales(variables)
                bounds = node.bounds(variables)

                def residual(values, duration):
                    trial = {**node.state, **values}
                    evaluated = node.trial_state(trial, adjacent, self.state.node)
                    return node.residual(
                        trial, node.state, evaluated, adjacent, duration, heat_rate
                    )

                try:
                    base = residual(variables, 0.0)
                    forcing = residual(variables, 1.0) - base
                    columns = []
                    for name in names:
                        delta = 1.0e-6 * scales[name]
                        low, high = bounds[name]
                        if variables[name] + delta > high:
                            delta = -delta
                        if not low <= variables[name] + delta <= high:
                            raise ValueError("No admissible predictor perturbation")
                        perturbed = {**variables, name: variables[name] + delta}
                        columns.append((residual(perturbed, 0.0) - base) / delta)
                    rates = np.linalg.solve(np.column_stack(columns), -forcing)
                    if np.all(np.isfinite(rates)):
                        predicted.node[node_id].update(
                            {name: variables[name] + dt * rate
                             for name, rate in zip(names, rates)}
                        )
                except (ValueError, np.linalg.LinAlgError):
                    # Prediction is optional; implicit event bracketing is the fallback.
                    continue
            return self._event_values(predicted)
        finally:
            self._restore(snapshot)

    def _event_step(self, dt, bcs, heat_rate):
        """Return the earliest event-aligned step and its converged candidate."""

        start_values = self._event_values(self.state)
        lower = 0.0
        lower_values = start_values
        lower_result = None
        upper = float(dt)
        upper_values = None
        upper_result = None
        last_failure = None
        predicted = self._predict_event_values(dt, bcs, heat_rate)
        estimates = [
            dt * start / (start - predicted[event])
            for event, start in start_values.items()
            if start > 0.0 and predicted.get(event, start) <= 0.0
        ]
        step = max(np.finfo(float).eps * dt, min(estimates, default=upper))
        if dt - step <= self.event_time_tolerance:
            step = float(dt)
        while True:
            try:
                result = self._solve(step, bcs, heat_rate, commit=False)
            except (LookupBoundsError, OperatingInfeasible):
                raise
            except (ValueError, RuntimeError) as error:
                last_failure = error
                upper = step
                upper_values = None
                upper_result = None
                if upper <= self.event_time_tolerance and lower_result is None:
                    raise
            else:
                values = self._event_values(result["state"])
                estimates = [
                    (step * start / (start - values[event]), event)
                    for event, start in start_values.items()
                    if start > 0.0 and values.get(event, start) < start
                ]
                if estimates:
                    estimate, event = min(estimates)
                    if abs(estimate - step) <= self.event_time_tolerance:
                        return step, result, event
                if self._crossed(start_values, values):
                    upper, upper_values = step, values
                    upper_result = result
                else:
                    lower, lower_values, lower_result = step, values, result
                    if step == dt:
                        return step, result, None

            if upper - lower <= self.event_time_tolerance:
                if upper_result is not None:
                    crossed = self._crossed(start_values, upper_values)
                    event = min(
                        crossed,
                        key=lambda key: lower_values[key] / (lower_values[key] - upper_values[key]),
                    )
                    return upper, upper_result, event
                if lower_result is None:
                    raise RuntimeError("Cannot resolve a fluid event within the time tolerance")
                # A failed upper trial is only an event if a measured margin
                # projects to zero within the event-time tolerance of this state.
                projected = [
                    (lower * start / (start - lower_values[event]), event)
                    for event, start in start_values.items()
                    if start > 0.0 and lower_values.get(event, start) < start
                ]
                near = [item for item in projected if item[0] <= upper + self.event_time_tolerance]
                if not near and lower <= self.event_time_tolerance:
                    raise RuntimeError(
                        f"Fluid network cannot advance at t={self.time:.9g} s "
                        f"within event_time_tolerance={self.event_time_tolerance:g} s "
                        "without a resolvable event"
                    ) from last_failure
                return lower, lower_result, min(near)[1] if near else None

            if upper_values is not None:
                estimates = [
                    lower + (upper - lower) * value / (value - upper_values[event])
                    for event, value in lower_values.items()
                    if value > 0.0 and upper_values.get(event, value) <= 0.0
                ]
            else:
                estimates = [
                    lower * start / (start - lower_values[event])
                    for event, start in start_values.items()
                    if start > 0.0 and lower_values.get(event, start) < start
                ]
            estimate = min(estimates, default=(lower + upper) / 2.0)
            # A failed upper trial must shrink geometrically; clipping a remote
            # event estimate by a fixed epsilon can otherwise take millions of
            # retries. Keep fast interpolation near a measured event crossing.
            margin = 0.1 * (upper - lower)
            if last_failure is None or upper_values is not None:
                margin = min(self.event_time_tolerance * 0.1, margin)
            step = float(np.clip(estimate, lower + margin, upper - margin))

    def _apply_event(self, event: tuple) -> bool:
        kind, component_id, name = event
        component = (
            self.nodes[component_id]
            if kind == "node"
            else self.branches[component_id]
        )
        was_open = getattr(component, "is_open", None)
        if kind == "node" and name.startswith("triple_point:"):
            self.frozen = True
            for node in self.nodes.values():
                if isinstance(node, CombustorComponent):
                    node.mode = "shutdown"
                    node.shutdown_reason = node.shutdown_reason or "triple_point"
                    node.state.update(mode="shutdown", shutdown_reason=node.shutdown_reason,
                                      cstar=0.0, Cf=0.0, mdot_oxidizer=0.0, mdot_fuel=0.0)
            self._freeze_shutdown(self.state.node, self.state.branches, {}, commit=True)
            changed = True
        else:
            changed = component.apply_event(name)
        if changed:
            key = f"{kind}:{component_id}:{name}"
            self.event_counts[key] = self.event_counts.get(key, 0) + 1
            record = {"time_s": self.time, "kind": kind, "component": component_id,
                      "event": name, "count": self.event_counts[key]}
            if was_open is not None:
                record.update(was_open=bool(was_open), is_open=bool(component.is_open))
            self.events.append(record)
        if changed and kind == "node" and not self.frozen:
            evaluated = component.trial_state(component.state, [], {})
            component.evaluated = component.output_state(evaluated)
            self.state.nodes[component_id] = component.evaluated
        return changed

    @staticmethod
    def _mutable_state(component, names):
        return {
            name: deepcopy(getattr(component, name))
            for name in names
            if hasattr(component, name)
        }

    def _snapshot(self):
        return (
            deepcopy(self.state),
            {
                node_id: (
                    self._mutable_state(
                        node,
                        (
                            "state",
                            "evaluated",
                            "model",
                            "mode",
                            "shutdown_reason",
                            "_condensation_armed",
                        ),
                    ),
                    node.definition.get("fluid"),
                    "fluid" in node.definition,
                )
                for node_id, node in self.nodes.items()
            },
            {
                branch_id: self._mutable_state(
                    branch,
                    (
                        "state",
                        "model",
                        "fluid",
                        "enabled",
                        "is_open",
                        "_open_state",
                    ),
                )
                for branch_id, branch in self.branches.items()
            },
            self.time, deepcopy(self.events), self.event_counts.copy(), self.frozen,
        )

    def _restore(self, snapshot) -> None:
        (self.state, node_states, branch_states,
         self.time, self.events, self.event_counts, self.frozen) = snapshot
        for node_id, (state, fluid, had_fluid) in node_states.items():
            self.nodes[node_id].__dict__.update(state)
            if had_fluid:
                self.nodes[node_id].definition["fluid"] = fluid
            else:
                self.nodes[node_id].definition.pop("fluid", None)
        for branch_id, state in branch_states.items():
            self.branches[branch_id].__dict__.update(state)

    def update(
        self,
        dt: Optional[float] = None,
        bcs: Optional[Dict[str, Dict[str, Any]]] = None,
        heat_rate: Optional[Dict[str, Any]] = None,
        commit: bool = True,
        axial_specific_force: float = 0.0,
    ) -> Dict[str, Any]:
        """Propagate the network and apply component mode transitions."""

        axial_specific_force = float(axial_specific_force)
        if not np.isfinite(axial_specific_force):
            raise ValueError("Axial specific force must be finite")
        for node in self.nodes.values():
            node.axial_specific_force = axial_specific_force

        if not commit:
            snapshot = self._snapshot()
            try:
                return self.update(
                    dt,
                    bcs,
                    heat_rate,
                    commit=True,
                    axial_specific_force=axial_specific_force,
                )
            finally:
                self._restore(snapshot)
        if dt is not None and (not np.isfinite(dt) or dt < 0.0):
            raise ValueError("Fluid-network dt must be finite and nonnegative")
        self.events = []
        margins = {}
        constraint_times = {}
        if self.frozen:
            self.time += dt or 0.0
            return {**self._event_output(self._output(self.state)),
                    "constraints": margins, "constraint_times": constraint_times}

        def active():
            chambers = [node_id for node_id, node in self.nodes.items() if isinstance(node, CombustorComponent)]
            return not chambers or any(self.state.nodes.get(key, {}).get("mode") != "shutdown" for key in chambers)

        def record(candidate, enabled=True):
            if self.constraint_monitor is not None and enabled:
                merge_margins(margins, self.constraint_monitor(candidate),
                              times=constraint_times, time=self.time)

        if dt is None:
            result = self._solve(dt, bcs, heat_rate, commit=True, detect_modes=True)
            record(result["state"])
            for event, value in self._event_values(self.state).items():
                if event[2].startswith("triple_point:") and value <= 0.0:
                    self._apply_event(event)
                    result = self._output(self.state)
                    break
            return {**self._event_output(result), "constraints": margins, "constraint_times": constraint_times}
        bcs = bcs or {}
        heat_rate = heat_rate or {}
        if not self.state.node:
            self._solve(None, bcs, heat_rate, commit=True, detect_modes=True)
        record(self.state, active())
        result = self._settle_transitions(bcs, heat_rate)
        record(self.state, active())

        start_time = self.time
        remaining = dt
        while remaining > 0.0 and not self.frozen and not (self.stop_at_shutdown and self._is_shutdown()):
            check = active()
            step, result, event = self._event_step(remaining, bcs, heat_rate)
            self._commit_candidate(result["state"], bcs)
            remaining -= step
            self.time = start_time + dt - remaining
            record(result["state"], check)

            if event is not None:
                settled = self._settle_transitions(
                    bcs,
                    heat_rate,
                    refresh=self._apply_event(event),
                )
                if settled is not None:
                    result = settled
                    record(result["state"], check)
        if result is None:
            result = self._solve(None, bcs, heat_rate, commit=True)
            record(result["state"], active())
        self.time = start_time + dt
        return {**self._event_output(result), "constraints": margins, "constraint_times": constraint_times}

    def _event_output(self, result):
        return {**result, "events": tuple(dict(event) for event in self.events),
                "event_counts": self.event_counts.copy()}

    def _apply_due_events(self) -> bool:
        changed = False
        due = [
            event
            for event, value in self._event_values(self.state).items()
            if value <= 0.0
        ]
        for event in due:
            changed = self._apply_event(event) or changed
            if self.frozen:
                break
        return changed

    def _apply_mode_transitions(self, branch_states=None, node_states=None) -> bool:
        changed = False
        branch_states = self.state.branches if branch_states is None else branch_states
        node_states = self.state.node if node_states is None else node_states
        for node in self.nodes.values():
            if not isinstance(node, CombustorComponent):
                continue
            inflows = self._adjacent(node, branch_states)
            mode_changed = node.update_mode(inflows)
            if mode_changed:
                changed = True
            phases = getattr(node.model, "phases", ())
            for branch_id in node.outgoing:
                branch = self.branches[branch_id]
                if isinstance(branch, NozzleComponent):
                    nozzle_changed = branch.set_mode(
                        node.mode == "combusting", phases
                    )
                    if nozzle_changed:
                        changed = True
                    if nozzle_changed and node.mode == "shutdown":
                        active_inflows = [
                            flow
                            for incidence, flow in iter_flows(inflows)
                            if incidence * int(flow["direction"]) > 0
                            and abs(float(flow["mdot"])) > self.flow_direction_tolerance
                        ]
                        branch.initialize_shutdown(
                            active_inflows,
                            float(node.state["P"]),
                            float(
                                node_states[node.definition["ambient_node"]]["P"]
                            ),
                        )
        return changed

    def _settle_transitions(self, bcs, heat_rate, refresh=False):
        """Solve between discrete events and their dependent mode changes."""

        if self.frozen:
            return self._output(self.state)
        if self.stop_at_shutdown and self._is_shutdown():
            return self._output(self.state)
        result = (
            self._solve(
                0.0,
                bcs,
                heat_rate,
                commit=True,
                detect_modes=True,
            )
            if refresh
            else None
        )
        while True:
            if self.stop_at_shutdown and self._is_shutdown():
                return result
            events_changed = self._apply_due_events()
            if self.frozen:
                return self._output(self.state)
            if events_changed:
                result = self._solve(
                    0.0,
                    bcs,
                    heat_rate,
                    commit=True,
                    detect_modes=True,
                )
            if self.stop_at_shutdown and self._is_shutdown():
                return result
            modes_changed = self._apply_mode_transitions()
            if modes_changed:
                result = self._solve(0.0, bcs, heat_rate, commit=True,
                                     detect_modes=self.stop_at_shutdown)
            if not events_changed and not modes_changed:
                return result
