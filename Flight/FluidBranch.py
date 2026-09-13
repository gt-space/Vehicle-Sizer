from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import numpy as np

from .FluidsDef import FluidsDef
from .FluidState import BranchState, FluidState


def port_pressure(node_state, port) -> float:
    """Return pressure at a node port, falling back to the bulk node pressure."""

    return float(node_state.get("port_pressure", {}).get(port, node_state["P"]))


class BranchModel:
    """Physics and solver-state interface used by a fluid branch."""

    phase = "unknown"
    requires_enthalpy = True

    def initial_state(self, branch) -> BranchState:
        return BranchState(
            flows={
                "main": {
                    "mdot": 0.0,
                    "direction": 1,
                    "fluid": FluidState(branch.fluid, self.phase),
                }
            }
        )

    def state(self, branch) -> Dict[str, float]:
        if not branch.enabled:
            return {}
        mdot = branch.state.mdot
        guess = float(branch.parameters.get("design_mdot", 1.0e-6))
        return {"mdot": mdot if mdot != 0.0 else guess}

    def scales(self, branch, state) -> Dict[str, float]:
        design = abs(float(branch.parameters.get("design_mdot", 0.0)))
        return {
            name: (
                1.0
                if name == "gas_area_fraction"
                else max(abs(float(value)), design, 1.0)
            )
            for name, value in state.items()
        }

    def evaluate(self, branch, state, node_state, source_fluids, directions):
        mdot = float(state.get("mdot", 0.0)) if branch.enabled else 0.0
        if len(source_fluids) != 1:
            raise ValueError(f"Branch '{branch.id}' requires one source fluid")
        fluid = next(iter(source_fluids.values()))
        result = BranchState(
            state={name: value for name, value in state.items() if name != "mdot"},
            flows={"main": {"mdot": mdot, "direction": directions["main"], "fluid": fluid}},
            dP=(
                port_pressure(node_state[branch.from_node], branch.from_port)
                - port_pressure(node_state[branch.to_node], branch.to_port)
            ),
            enabled=branch.enabled,
        )
        if self.requires_enthalpy:
            if "h" not in fluid:
                raise ValueError(
                    f"Branch '{branch.id}' source fluid '{fluid.fluid}' requires h"
                )
            result.metadata["h"] = fluid["h"]
        result.metadata.update(
            {
                name: fluid[name]
                for name in ("T", "rho", "R", "gamma")
                if name in fluid
            }
        )
        return result

    def residual(self, branch, state, node_state) -> np.ndarray:
        raise NotImplementedError

    def committed_state(self, branch, state: BranchState) -> BranchState:
        return BranchState(
            state=dict(state.state),
            flows=state.flows,
            dP=state.dP,
            enabled=state.enabled,
            metadata=state.metadata,
        )

    def total_mdot(self, state) -> float:
        return float(state.get("mdot", 0.0))


class IncompressibleLossModel(BranchModel):
    phase = "liquid"

    def residual(self, branch, state, node_state) -> np.ndarray:
        expected = FluidsDef.incompressible_mdot(
            branch.effective_cda(), state["rho"], state["dP"]
        )
        return np.array([(state["mdot"] - expected) / branch.flow_scale("mdot")])


class CompressibleLossModel(BranchModel):
    phase = "gas"

    def residual(self, branch, state, node_state) -> np.ndarray:
        direction = 1.0 if state["dP"] >= 0.0 else -1.0
        expected = direction * FluidsDef.compressible_mdot(
            branch.effective_cda(),
            max(node_state[branch.from_node]["P"], node_state[branch.to_node]["P"]),
            min(node_state[branch.from_node]["P"], node_state[branch.to_node]["P"]),
            state["T"],
            state["R"],
            state["gamma"],
        )
        return np.array([(state["mdot"] - expected) / branch.flow_scale("mdot")])


class PumpModel(BranchModel):
    phase = "liquid"

    def residual(self, branch, state, node_state) -> np.ndarray:
        mdot = state["mdot"]
        head_model = branch.parameters.get("head_model")
        head = head_model(mdot) if callable(head_model) else branch.parameters["dP"]
        loss = 0.0
        if branch.parameters.get("CdA") is not None:
            loss = np.sign(mdot) * (mdot / branch.parameters["CdA"]) ** 2
            loss /= 2.0 * state["rho"]
        return np.array([(state["dP"] + head - loss) / max(abs(float(head)), 1.0e5)])


class NozzleModel(BranchModel):
    """Shared base for combustion and twin-path nozzle equations."""

    requires_enthalpy = False


class CombustionNozzleModel(NozzleModel):
    phase = "gas"

    def residual(self, branch, state, node_state) -> np.ndarray:
        chamber = node_state[branch.from_node]
        expected = 0.0
        if state["dP"] > 0.0 and chamber["cstar"] > 0.0:
            expected = (
                branch.parameters["Cd"]
                * chamber["P"]
                * branch.parameters["At"]
                / chamber["cstar"]
            )
        return np.array([(state["mdot"] - expected) / branch.flow_scale("mdot")])


class TwinPathNozzleModel(NozzleModel):
    """Gas and incompressible-liquid flow sharing one physical throat."""

    def __init__(self, phases):
        self.phases = tuple(phases)

    def initial_state(self, branch) -> BranchState:
        state = {}
        if len(self.phases) == 2:
            state["gas_area_fraction"] = 0.5
        return BranchState(
            state=state,
            flows={
                phase: {
                    "mdot": 0.0,
                    "direction": 1,
                    "fluid": FluidState(branch.fluid, phase),
                }
                for phase in self.phases
            },
        )

    def state(self, branch) -> Dict[str, float]:
        if not branch.enabled:
            return {}
        variables = {
            f"mdot_{phase}": branch.state.phase_mdot(phase) or 1.0e-6
            for phase in self.phases
        }
        if len(self.phases) == 2:
            variables["gas_area_fraction"] = float(
                branch.state.state.get("gas_area_fraction", 0.5)
            )
        return variables

    def total_mdot(self, variables: Dict[str, float]) -> float:
        return sum(float(variables[f"mdot_{phase}"]) for phase in self.phases)

    def evaluate(self, branch, state, node_state, source_fluids, directions):
        flows = {}
        missing = []
        for phase in self.phases:
            sources = {
                name: fluid
                for name, fluid in source_fluids.items()
                if fluid.phase == phase
            }
            if not sources:
                missing.append(phase)
                continue
            target = float(state[f"mdot_{phase}"])
            share = target / len(sources)
            for name, fluid in sources.items():
                flows[name] = {
                    "mdot": share,
                    "direction": directions[phase],
                    "fluid": fluid,
                }
        return BranchState(
            state={name: value for name, value in state.items() if not name.startswith("mdot_")},
            flows=flows,
            dP=(
                port_pressure(node_state[branch.from_node], branch.from_port)
                - port_pressure(node_state[branch.to_node], branch.to_port)
            ),
            enabled=branch.enabled,
            metadata={"missing_phases": missing},
        )

    def residual(self, branch, state, node_state) -> np.ndarray:
        definition = branch.parameters
        chamber_pressure = node_state[branch.from_node]["P"]
        ambient_pressure = node_state[branch.to_node]["P"]
        area_fraction = float(state.get("gas_area_fraction", 1.0 if self.phases == ("gas",) else 0.0))
        area = {"gas": area_fraction, "liquid": 1.0 - area_fraction}
        equations = []
        thrust = 0.0
        for phase in self.phases:
            mdot = state.phase_mdot(phase)
            if phase in state.metadata["missing_phases"]:
                equations.append(mdot / branch.flow_scale(f"mdot_{phase}"))
                continue
            constituents = [flow for flow in state.flows.values() if flow["fluid"].phase == phase]
            phase_mdot = sum(float(flow["mdot"]) for flow in constituents)
            if phase_mdot == 0.0:
                raise ValueError(f"Nozzle '{branch.id}' has zero {phase} trial flow")
            capacity = 0.0
            for flow in constituents:
                fluid = flow["fluid"]
                if phase == "gas":
                    flux = FluidsDef.compressible_mass_flux(
                        chamber_pressure,
                        ambient_pressure,
                        fluid["T"],
                        fluid["R"],
                        fluid["gamma"],
                    )
                    velocity = FluidsDef.isentropic_velocity(
                        chamber_pressure,
                        ambient_pressure,
                        fluid["T"],
                        fluid["R"],
                        fluid["gamma"],
                    )
                else:
                    flux = np.sqrt(
                        max(2.0 * fluid["rho"] * state["dP"], 0.0)
                    )
                    velocity = flux / fluid["rho"]
                fraction = float(flow["mdot"]) / phase_mdot
                capacity += fraction * flux
                thrust += float(flow["mdot"]) * velocity
            expected = (
                definition["Cd"] * definition["At"] * area[phase] * capacity
            )
            equations.append((mdot - expected) / branch.flow_scale(f"mdot_{phase}"))
        state.state["gas_area_fraction"] = area_fraction
        state.metadata["thrust"] = thrust
        return np.asarray(equations)


class FluidBranch:
    """Solver branch holding flow state and an active physics model."""

    tracks_stream = True

    def __init__(self, branch_id: str, definition: Dict[str, Any], model: BranchModel):
        self.id = branch_id
        self.parameters = {
            name: value
            for name, value in definition.items()
            if name not in ("from", "to", "from_port", "to_port")
        }
        self.model = model
        self.from_node = str(definition.get("from", ""))
        self.to_node = str(definition.get("to", ""))
        self.fluid = str(definition["fluid"])
        self.from_port = definition.get("from_port", self.fluid)
        self.to_port = definition.get("to_port", self.fluid)
        self.enabled = True
        self.state = model.initial_state(self)

    def switch_model(self, model: BranchModel) -> None:
        self.model = model
        self.state = model.initial_state(self)

    def effective_cda(self) -> float:
        return float(self.parameters["CdA"])

    def solver_state(self) -> Dict[str, float]:
        return self.model.state(self)

    def scales(self, state) -> Dict[str, float]:
        return self.model.scales(self, state)

    def total_mdot(self, state) -> float:
        return state.mdot if isinstance(state, BranchState) else self.model.total_mdot(state)

    def evaluate(self, state, node_state, source_fluids, directions) -> BranchState:
        result = self.model.evaluate(self, state, node_state, source_fluids, directions)
        if len(result.flows) == 1:
            self.fluid = next(iter(result.flows.values()))["fluid"].fluid
        return result

    def residual(self, state, node_state) -> np.ndarray:
        if not self.enabled:
            return np.empty(0)
        return self.model.residual(self, state, node_state)

    def event_values(self, node_state) -> Dict[str, float]:
        return {}

    def apply_event(self, name: str) -> bool:
        raise ValueError(f"Branch '{self.id}' has no event {name!r}")

    def commit(self, state: BranchState) -> None:
        self.state = self.model.committed_state(self, state)

    def close(self) -> None:
        self.enabled = False
        self.state.state = {name: 0.0 for name in self.state.state}
        for flow in self.state.flows.values():
            flow["mdot"] = 0.0

    def flow_scale(self, name: str) -> float:
        return max(
            abs(float(self.state.state.get(name, 0.0))),
            abs(float(self.parameters.get("design_mdot", 0.0))),
            1.0,
        )


class LossComponent(FluidBranch):
    """Passive loss selecting compressible or incompressible physics."""

    tracks_stream = True

    def __init__(self, branch_id: str, definition: Dict[str, Any]):
        super().__init__(branch_id, definition, IncompressibleLossModel())

    def evaluate(self, state, node_state, source_fluids, directions):
        if len(source_fluids) != 1:
            raise ValueError(f"Loss branch '{self.id}' requires one source fluid")
        phase = next(iter(source_fluids.values())).phase
        if phase == "liquid":
            model = IncompressibleLossModel
        elif phase == "gas":
            model = CompressibleLossModel
        else:
            raise ValueError(f"Loss branch '{self.id}' cannot transport phase '{phase}'")
        if not isinstance(self.model, model):
            self.model = model()
        return super().evaluate(state, node_state, source_fluids, directions)


class BangBangValveComponent(LossComponent):
    """Pressure-controlled gas valve with persistent hysteresis state."""

    def __init__(self, branch_id: str, definition: Dict[str, Any]):
        FluidBranch.__init__(self, branch_id, definition, CompressibleLossModel())
        target = float(definition["target_pressure"])
        band = float(definition["pressure_band"])
        if not 0.0 < band < target:
            raise ValueError(
                f"Bang-bang valve '{branch_id}' requires 0 < pressure_band "
                "< target_pressure"
            )
        initially_open = definition["initially_open"]
        if not isinstance(initially_open, bool):
            raise ValueError("Bang-bang initially_open must be boolean")
        self.is_open = initially_open
        self._open_state = deepcopy(self.state)

    def effective_cda(self) -> float:
        return float(self.parameters["CdA"])

    def solver_state(self) -> Dict[str, float]:
        return super().solver_state() if self.is_open else {}

    def evaluate(self, state, node_state, source_fluids, directions):
        result = super().evaluate(state, node_state, source_fluids, directions)
        result.metadata["is_open"] = self.is_open
        return result

    def residual(self, state, node_state) -> np.ndarray:
        return super().residual(state, node_state) if self.is_open else np.empty(0)

    def event_values(self, node_state) -> Dict[str, float]:
        pressure = float(node_state[self.to_node]["P"])
        target = float(self.parameters["target_pressure"])
        band = float(self.parameters["pressure_band"])
        margin = target + band - pressure if self.is_open else pressure - target + band
        return {"switch": margin}

    def apply_event(self, name: str) -> bool:
        if name != "switch":
            return super().apply_event(name)
        self._set_open(not self.is_open)
        return True

    def _set_open(self, is_open: bool) -> None:
        if self.is_open and not is_open:
            self._open_state = deepcopy(self.state)
        self.is_open = is_open
        self.state = (
            deepcopy(self._open_state)
            if self.is_open
            else BranchState(
                state={name: 0.0 for name in self.state.state},
                flows={
                    name: {**flow, "mdot": 0.0}
                    for name, flow in self.state.flows.items()
                },
            )
        )

    def update_control(self, tank_pressure: float) -> bool:
        target = float(self.parameters["target_pressure"])
        band = float(self.parameters["pressure_band"])
        next_state = self.is_open
        if self.is_open and tank_pressure >= target + band:
            next_state = False
        elif not self.is_open and tank_pressure <= target - band:
            next_state = True
        if next_state == self.is_open:
            return False
        self._set_open(next_state)
        return True


class PumpComponent(FluidBranch):
    tracks_stream = True

    def __init__(self, branch_id: str, definition: Dict[str, Any]):
        super().__init__(branch_id, definition, PumpModel())


class NozzleComponent(FluidBranch):
    tracks_stream = True

    def __init__(self, branch_id: str, definition: Dict[str, Any]):
        super().__init__(branch_id, definition, CombustionNozzleModel())

    def set_mode(self, combusting: bool, phases=()) -> bool:
        model = CombustionNozzleModel() if combusting else TwinPathNozzleModel(phases)
        if type(self.model) is type(model) and getattr(self.model, "phases", ()) == tuple(phases):
            return False
        self.switch_model(model)
        return True

    def evaluate(self, state, node_state, source_fluids, directions):
        if not source_fluids:
            raise ValueError(f"Nozzle branch '{self.id}' requires a source fluid")
        if isinstance(self.model, CombustionNozzleModel) and (
            len(source_fluids) != 1 or next(iter(source_fluids.values())).phase != "gas"
        ):
            raise ValueError(f"Combustion nozzle '{self.id}' requires one gas stream")
        return super().evaluate(state, node_state, source_fluids, directions)
