from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .FluidsDef import FluidsDef
from .FluidNode import FlowConn


class BranchModel:
    """Physics and solver-state interface used by a fluid branch."""

    phase = "unknown"
    requires_enthalpy = True

    def initial_state(self, branch) -> Dict[str, float]:
        return {"mdot": 0.0}

    def state_variables(self, branch) -> Dict[str, float]:
        if not branch.enabled:
            return {}
        mdot = float(branch.state["mdot"])
        return {"mdot": mdot if mdot != 0.0 else 1.0e-6}

    def flow_state(
        self, branch, variables: Dict[str, float], inlet: FlowConn | None = None
    ) -> Dict[str, Any]:
        mdot = float(variables.get("mdot", 0.0)) if branch.enabled else 0.0
        if inlet is None:
            components = {branch.fluid: {"phase": self.phase, "mdot": mdot}}
        else:
            if len(inlet.components) != 1:
                raise ValueError(f"Branch '{branch.id}' requires one inlet fluid")
            fluid, source = next(iter(inlet.components.items()))
            components = {fluid: {**source, "mdot": mdot}}
        fluid, component = next(iter(components.items()))
        return {
            "mdot": mdot,
            "fluid": fluid,
            "phase": component["phase"],
            "components": components,
            "enabled": branch.enabled,
        }

    def evaluate(self, branch, variables, node_state, inlet=None):
        state = self.flow_state(branch, variables, inlet)
        definition = branch.definition
        state["dP"] = (
            node_state[definition["from"]]["P"]
            - node_state[definition["to"]]["P"]
        )
        if self.requires_enthalpy:
            component = state["components"][state["fluid"]]
            if "h" not in component:
                raise ValueError(
                    f"Branch '{branch.id}' inlet fluid '{state['fluid']}' requires h"
                )
            state["h"] = component["h"]
        state.update(
            {
                name: state["components"][state["fluid"]][name]
                for name in ("T", "rho", "R", "gamma")
                if name in state["components"][state["fluid"]]
            }
        )
        return state

    def residual(self, branch, state, node_state) -> np.ndarray:
        raise NotImplementedError

    def commit(self, branch, state: Dict[str, Any]) -> Dict[str, float]:
        return {name: float(state[name]) for name in self.initial_state(branch)}

    def total_mdot(self, variables: Dict[str, float]) -> float:
        return float(variables.get("mdot", 0.0))


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
        definition = branch.definition
        expected = direction * FluidsDef.compressible_mdot(
            branch.effective_cda(),
            max(node_state[definition["from"]]["P"], node_state[definition["to"]]["P"]),
            min(node_state[definition["from"]]["P"], node_state[definition["to"]]["P"]),
            state["T"],
            state["R"],
            state["gamma"],
        )
        return np.array([(state["mdot"] - expected) / branch.flow_scale("mdot")])


class PumpModel(BranchModel):
    phase = "liquid"

    def residual(self, branch, state, node_state) -> np.ndarray:
        mdot = state["mdot"]
        head_model = branch.definition.get("head_model")
        head = head_model(mdot) if callable(head_model) else branch.definition["dP"]
        loss = 0.0
        if branch.definition.get("CdA") is not None:
            loss = np.sign(mdot) * (mdot / branch.definition["CdA"]) ** 2
            loss /= 2.0 * state["rho"]
        return np.array([(state["dP"] + head - loss) / max(abs(float(head)), 1.0e5)])


class NozzleModel(BranchModel):
    """Shared base for combustion and twin-path nozzle equations."""

    requires_enthalpy = False


class CombustionNozzleModel(NozzleModel):
    phase = "gas"

    def residual(self, branch, state, node_state) -> np.ndarray:
        chamber = node_state[branch.definition["from"]]
        expected = 0.0
        if state["dP"] > 0.0 and chamber["cstar"] > 0.0:
            expected = (
                branch.definition["Cd"]
                * chamber["P"]
                * branch.definition["At"]
                / chamber["cstar"]
            )
        return np.array([(state["mdot"] - expected) / branch.flow_scale("mdot")])


class TwinPathNozzleModel(NozzleModel):
    """Gas and incompressible-liquid flow sharing one physical throat."""

    def __init__(self, phases):
        self.phases = tuple(phases)

    def initial_state(self, branch) -> Dict[str, float]:
        state = {f"mdot_{phase}": 0.0 for phase in self.phases}
        if len(self.phases) == 2:
            state["gas_area_fraction"] = 0.5
        return state

    def state_variables(self, branch) -> Dict[str, float]:
        if not branch.enabled:
            return {}
        variables = dict(branch.state)
        for name in tuple(variables):
            if name.startswith("mdot_") and variables[name] == 0.0:
                variables[name] = 1.0e-6
        return variables

    def total_mdot(self, variables: Dict[str, float]) -> float:
        return sum(float(variables[f"mdot_{phase}"]) for phase in self.phases)

    def flow_state(
        self, branch, variables: Dict[str, float], inlet: FlowConn | None = None
    ) -> Dict[str, Any]:
        if inlet is None:
            return {
                **variables,
                "mdot": self.total_mdot(variables),
                "components": {},
                "enabled": branch.enabled,
            }
        components = {}
        for phase in self.phases:
            sources = {
                fluid: state
                for fluid, state in inlet.components.items()
                if state["phase"] == phase
            }
            if not sources:
                raise ValueError(f"Nozzle '{branch.id}' has no {phase} inlet constituent")
            target = float(variables[f"mdot_{phase}"])
            source_total = sum(max(float(item.get("mdot", 0.0)), 0.0) for item in sources.values())
            if len(sources) > 1 and source_total <= 0.0:
                raise ValueError(f"Nozzle '{branch.id}' cannot split zero {phase} inlet flow")
            for fluid, source in sources.items():
                fraction = 1.0 if len(sources) == 1 else max(float(source["mdot"]), 0.0) / source_total
                components[fluid] = {**source, "mdot": target * fraction}
        mdot = self.total_mdot(variables) if branch.enabled else 0.0
        phase = (
            self.phases[0]
            if len(self.phases) == 1
            else "twin_path" if self.phases else "none"
        )
        return {
            **variables,
            "mdot": mdot,
            "phase": phase,
            "components": components,
            "enabled": branch.enabled,
        }

    def evaluate(self, branch, variables, node_state, inlet=None):
        if inlet is None:
            raise ValueError(f"Twin-path nozzle '{branch.id}' requires an inlet")
        state = self.flow_state(branch, variables, inlet)
        definition = branch.definition
        state["dP"] = node_state[definition["from"]]["P"] - node_state[definition["to"]]["P"]
        return state

    def residual(self, branch, state, node_state) -> np.ndarray:
        definition = branch.definition
        chamber_pressure = node_state[definition["from"]]["P"]
        ambient_pressure = node_state[definition["to"]]["P"]
        area_fraction = float(state.get("gas_area_fraction", 1.0 if self.phases == ("gas",) else 0.0))
        area = {"gas": area_fraction, "liquid": 1.0 - area_fraction}
        equations = []
        thrust = 0.0
        for phase in self.phases:
            mdot = state[f"mdot_{phase}"]
            constituents = [
                item
                for item in state["components"].values()
                if item["phase"] == phase
            ]
            phase_mdot = sum(float(item["mdot"]) for item in constituents)
            if phase_mdot == 0.0:
                raise ValueError(f"Nozzle '{branch.id}' has zero {phase} trial flow")
            capacity = 0.0
            for fluid in constituents:
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
                fraction = float(fluid["mdot"]) / phase_mdot
                capacity += fraction * flux
                thrust += float(fluid["mdot"]) * velocity
            expected = (
                definition["Cd"] * definition["At"] * area[phase] * capacity
            )
            equations.append((mdot - expected) / branch.flow_scale(f"mdot_{phase}"))
        state["gas_area_fraction"] = area_fraction
        state["thrust"] = thrust
        return np.asarray(equations)


class FluidBranch:
    """Solver branch holding flow state and an active physics model."""

    tracks_stream = True

    def __init__(self, branch_id: str, definition: Dict[str, Any], model: BranchModel):
        self.id = branch_id
        self.definition = definition
        self.model = model
        self.fluid = str(definition["fluid"])
        self.enabled = True
        self.state = model.initial_state(self)

    def switch_model(self, model: BranchModel) -> None:
        self.model = model
        self.state = model.initial_state(self)

    def effective_cda(self) -> float:
        return float(self.definition["CdA"])

    def state_variables(self) -> Dict[str, float]:
        return self.model.state_variables(self)

    def flow_state(self, variables: Dict[str, float], inlet: FlowConn | None = None) -> Dict[str, Any]:
        return self.model.flow_state(self, variables, inlet)

    def total_mdot(self, variables: Dict[str, float]) -> float:
        return self.model.total_mdot(variables)

    def evaluate(self, variables, node_state, inlet: FlowConn | None = None) -> Dict[str, Any]:
        if inlet is not None and len(inlet.components) == 1:
            self.fluid = inlet.fluid
        return self.model.evaluate(self, variables, node_state, inlet)

    def residual(self, state, node_state) -> np.ndarray:
        if not self.enabled:
            return np.empty(0)
        return self.model.residual(self, state, node_state)

    def commit(self, state: Dict[str, Any]) -> None:
        self.state = self.model.commit(self, state)

    def close(self) -> None:
        self.enabled = False
        self.state = {name: 0.0 for name in self.state}

    def flow_scale(self, name: str) -> float:
        return max(abs(float(self.state.get(name, 0.0))), 1.0)


class LossComponent(FluidBranch):
    """Passive loss selecting compressible or incompressible physics."""

    tracks_stream = True

    def __init__(self, branch_id: str, definition: Dict[str, Any]):
        super().__init__(branch_id, definition, IncompressibleLossModel())

    def evaluate(self, variables, node_state, inlet=None):
        if inlet is None or len(inlet.components) != 1:
            raise ValueError(f"Loss branch '{self.id}' requires one inlet fluid")
        if inlet.phase == "liquid":
            model = IncompressibleLossModel
        elif inlet.phase == "gas":
            model = CompressibleLossModel
        else:
            raise ValueError(f"Loss branch '{self.id}' cannot transport phase '{inlet.phase}'")
        if not isinstance(self.model, model):
            self.model = model()
        return super().evaluate(variables, node_state, inlet)


class ValveComponent(LossComponent):
    """Bang-bang valve applying its duty cycle to the selected loss model."""

    def __init__(self, branch_id: str, definition: Dict[str, Any]):
        FluidBranch.__init__(self, branch_id, definition, CompressibleLossModel())

    def effective_cda(self) -> float:
        return float(self.definition["CdA"] * self.definition["duty_cycle"])


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

    def evaluate(self, variables, node_state, inlet=None):
        if inlet is None:
            raise ValueError(f"Nozzle branch '{self.id}' requires an inlet FlowConn")
        if isinstance(self.model, CombustionNozzleModel) and (
            len(inlet.components) != 1 or inlet.phase != "gas"
        ):
            raise ValueError(f"Combustion nozzle '{self.id}' requires one gas stream")
        return super().evaluate(variables, node_state, inlet)
