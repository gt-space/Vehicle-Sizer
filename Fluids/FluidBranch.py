"""Quasi-steady momentum/pressure-flow closures for a SUNDIALS DAE.

Branches have parameters, connectivity and discrete modes, but no integrated or
committed flow state. The network owns y, ydot, initialization and acceptance.
All variables below are algebraic; line inertia is not part of this model.
Equation implementations are local, not adapters around the old fluids solver.
"""

from copy import deepcopy
from math import copysign, isfinite, sqrt

import numpy as np

from .FluidState import BranchState
from errors import TrialDomainError


def _port_pressure(node, port):
    pressure = float(node.get("port_pressure", {}).get(port, node["P"]))
    if not isfinite(node["P"]) or not isfinite(pressure):
        raise TrialDomainError("Branch requires finite node and port pressures")
    return pressure


def _gas_flux(upstream, downstream, fluid):
    """Isentropic gas mass flux [kg/(m² s)], including choking."""
    temperature, gas_constant, gamma = fluid["T"], fluid["R"], fluid["gamma"]
    if (not all(isfinite(v) for v in (upstream, downstream, temperature, gas_constant, gamma))
            or upstream <= 0 or downstream < 0 or temperature <= 0 or gas_constant <= 0 or gamma <= 1):
        raise TrialDomainError("Gas flow requires finite physical absolute pressures, T, R and gamma")
    if downstream >= upstream:
        return 0.0
    ratio = downstream / upstream
    critical = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    if ratio <= critical:
        factor = sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    else:
        factor = sqrt(max(0.0, 2 * gamma / (gamma - 1)
                          * (ratio ** (2 / gamma) - ratio ** ((gamma + 1) / gamma))))
    return upstream * factor / sqrt(gas_constant * temperature)


def _liquid_flux(pressure_drop, fluid):
    """Signed incompressible mass flux [kg/(m² s)]."""
    if not isfinite(pressure_drop) or not isfinite(fluid["rho"]) or fluid["rho"] <= 0:
        raise TrialDomainError("Liquid flow requires finite pressure drop and positive finite density")
    return copysign(sqrt(2 * fluid["rho"] * abs(pressure_drop)), pressure_drop)


class FluidBranch:
    """Base contract: fixed-mode trial data and normalized algebraic residuals."""

    requires_enthalpy = True
    differential_variable_names = ()
    equation_names = ("flow",)

    def __init__(self, branch_id, definition, *, phase="liquid"):
        self.id = branch_id
        self.parameters = deepcopy(definition)
        self.from_node, self.to_node = definition["from"], definition["to"]
        self.from_port = definition.get("from_port", definition.get("fluid"))
        self.to_port = definition.get("to_port", definition.get("fluid"))
        self.enabled = True
        self.set_phase(phase)
        design_flow = float(definition.get("design_mdot", 0.0))
        if not isfinite(design_flow):
            raise ValueError("design_mdot must be finite")
        self.flow_scale = max(abs(design_flow), 1.0)

    @property
    def active(self):
        return self.enabled

    @property
    def variable_names(self):
        """Names/order to place in the network's algebraic part of y."""
        return ("mdot",)

    def set_phase(self, phase):
        if phase not in {"liquid", "gas"}:
            raise ValueError(f"Unsupported phase: {phase}")
        changed = getattr(self, "phase", None) != phase
        self.phase = phase
        return changed

    def set_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        changed = self.enabled != enabled
        self.enabled = enabled
        return changed

    def pressures(self, node_states_by_id):
        return (_port_pressure(node_states_by_id[self.from_node], self.from_port),
                _port_pressure(node_states_by_id[self.to_node], self.to_port))

    def _positive(self, name):
        value = float(self.parameters[name])
        if not isfinite(value) or value <= 0:
            raise ValueError(f"Branch '{self.id}' requires finite {name} > 0")
        return value

    def _check_trial_values(self, trial_values):
        if set(trial_values) != set(self.variable_names):
            raise ValueError(f"Branch '{self.id}' requires variables {self.variable_names}")
        if not all(isfinite(v) for v in trial_values.values()):
            raise TrialDomainError(f"Branch '{self.id}' has non-finite trial values")

    def evaluate(self, trial_values, node_states_by_id, source_fluids, directions):
        """Build a BranchState from local y values and endpoint/donor records.

        trial_values is keyed by variable_names. node_states_by_id supplies
        endpoint pressures; source_fluids and directions specify donor routing.
        This does not solve for flow or modify input/persistent records.
        """
        self._check_trial_values(trial_values)
        if not self.active:
            return BranchState(trial_values=trial_values, enabled=False,
                               properties={"thrust": 0.0, "pump_active": False})
        if len(source_fluids) != 1:
            raise ValueError(f"Branch '{self.id}' requires one source fluid")
        fluid = next(iter(source_fluids.values()))
        if fluid.phase != self.phase:
            raise ValueError(f"Branch '{self.id}' phase changed; apply the transition between solves")
        if self.requires_enthalpy and "h" not in fluid:
            raise ValueError(f"Branch '{self.id}' source fluid requires h")
        p_from, p_to = self.pressures(node_states_by_id)
        return BranchState(
            trial_values=trial_values,
            flows={"main": {"mdot": float(trial_values["mdot"]),
                            "direction": directions["main"], "fluid": fluid}},
            dP=p_from - p_to,
            properties={name: fluid[name] for name in ("h", "T", "rho", "R", "gamma")
                      if name in fluid and (name != "h" or self.requires_enthalpy)},
        )

    def residual(self, branch_state, node_states_by_id):
        """Return pressure/flow equation errors using evaluate's BranchState."""
        if not self.active:
            return np.array([branch_state.trial_values["mdot"] / self.flow_scale])
        return np.asarray(self._residual(branch_state, node_states_by_id), dtype=float)

    def _residual(self, branch_state, node_states_by_id):
        raise NotImplementedError("Select a branch with a pressure-flow law")

    def event_values(self, node_states_by_id):
        return {}


class LossComponent(FluidBranch):
    """Restriction with explicit liquid/gas mode and bidirectional flow."""

    def __init__(self, branch_id, definition, *, phase="liquid"):
        super().__init__(branch_id, definition, phase=phase)
        self.cda = self._positive("CdA")

    def mass_flow(self, branch_state, node_states_by_id):
        fluid = next(iter(branch_state.flows.values()))["fluid"]
        if self.phase == "liquid":
            return self.cda * _liquid_flux(branch_state.dP, fluid)
        # Preserve the original gas law: bulk pressures set capacity, while
        # the port pressure difference sets its sign.
        p_from = node_states_by_id[self.from_node]["P"]
        p_to = node_states_by_id[self.to_node]["P"]
        return copysign(self.cda * _gas_flux(max(p_from, p_to), min(p_from, p_to), fluid), branch_state.dP)

    def _residual(self, branch_state, node_states_by_id):
        return [(branch_state.mdot - self.mass_flow(branch_state, node_states_by_id)) / self.flow_scale]


class PumpComponent(FluidBranch):
    """Liquid pressure rise minus loss; passive gas restriction after dryout."""

    def __init__(self, branch_id, definition, *, phase="liquid"):
        super().__init__(branch_id, definition, phase=phase)
        self.gas_cda = self._positive("gas_CdA")
        self.loss_cda = self._positive("CdA") if definition.get("CdA") is not None else None
        self.head_model = definition.get("head_model")
        if self.head_model is not None and not callable(self.head_model):
            raise ValueError("head_model must be callable")
        self.design_head = self._positive("dP")

    def evaluate(self, trial_values, node_states_by_id, source_fluids, directions):
        branch_state = super().evaluate(trial_values, node_states_by_id, source_fluids, directions)
        branch_state.properties["pump_active"] = self.active and self.phase == "liquid"
        return branch_state

    def _residual(self, branch_state, node_states_by_id):
        fluid = next(iter(branch_state.flows.values()))["fluid"]
        if self.phase == "gas":
            p_from = node_states_by_id[self.from_node]["P"]
            p_to = node_states_by_id[self.to_node]["P"]
            expected = copysign(self.gas_cda * _gas_flux(max(p_from, p_to), min(p_from, p_to), fluid), branch_state.dP)
            return [(branch_state.mdot - expected) / self.flow_scale]
        head = self.head_model(branch_state.mdot) if self.head_model else self.design_head
        if not isfinite(head) or not isfinite(fluid["rho"]) or fluid["rho"] <= 0:
            raise TrialDomainError("Pump requires finite head and positive finite liquid density")
        loss = copysign((branch_state.mdot / self.loss_cda) ** 2 / (2 * fluid["rho"]), branch_state.mdot) if self.loss_cda else 0.0
        return [(branch_state.dP + head - loss) / max(abs(float(head)), 1e5)]


class RegulatorComponent(LossComponent):
    """Ideal non-relieving regulator: closed, regulating, or capacity-limited.

    The projected equation preserves the old idealization. Its zero-time flow
    ambiguity and consistent initialization must be addressed by the network.
    """

    def __init__(self, branch_id, definition):
        super().__init__(branch_id, definition, phase="gas")
        self.target = self._positive("target_pressure")

    def set_phase(self, phase):
        if phase != "gas":
            raise ValueError("Regulator requires gas")
        return super().set_phase(phase)

    def evaluate(self, trial_values, node_states_by_id, source_fluids, directions):
        branch_state = super().evaluate(trial_values, node_states_by_id, source_fluids, directions)
        capacity = max(0.0, self.mass_flow(branch_state, node_states_by_id)) if self.active else 0.0
        branch_state.properties.update(max_mdot=capacity,
            opening_fraction=float(np.clip(branch_state.mdot / capacity, 0, 1)) if capacity else 0.0,
            target_pressure=self.target, pressure_error=node_states_by_id[self.to_node]["P"] - self.target)
        return branch_state

    def _residual(self, branch_state, node_states_by_id):
        q = branch_state.mdot / self.flow_scale
        error = (self.target - node_states_by_id[self.to_node]["P"]) / self.target
        capacity = branch_state.properties["max_mdot"] / self.flow_scale
        return [q - np.clip(q + error, 0.0, capacity)]


class BangBangValveComponent(LossComponent):
    def __init__(self, branch_id, definition):
        super().__init__(branch_id, definition, phase="gas")
        self.target = self._positive("target_pressure")
        self.band = self._positive("pressure_band")
        if self.band >= self.target:
            raise ValueError("pressure_band must be smaller than target_pressure")
        self.set_open(definition["initially_open"])

    def set_phase(self, phase):
        if phase != "gas":
            raise ValueError("Bang-bang valve requires gas")
        return super().set_phase(phase)

    @property
    def active(self):
        return self.enabled and self.is_open

    def set_open(self, is_open):
        if not isinstance(is_open, bool):
            raise ValueError("is_open must be boolean")
        changed = getattr(self, "is_open", None) != is_open
        self.is_open = is_open
        return changed

    def evaluate(self, trial_values, node_states_by_id, source_fluids, directions):
        branch_state = super().evaluate(trial_values, node_states_by_id, source_fluids, directions)
        branch_state.properties["is_open"] = self.is_open
        return branch_state

    def event_values(self, node_states_by_id):
        """Active threshold, crossed in the negative direction; never toggles."""
        if not self.enabled:
            return {}
        p = node_states_by_id[self.to_node]["P"]
        return {"switch": self.target + self.band - p if self.is_open else p - self.target + self.band}


class NozzleComponent(FluidBranch):
    """Reacting throat or nonreacting gas/liquid discharge sharing one throat."""

    requires_enthalpy = False

    def __init__(self, branch_id, definition):
        super().__init__(branch_id, definition, phase="gas")
        self.throat_area = self._positive("At") * self._positive("Cd")
        self.set_mode(True)

    def set_phase(self, phase):
        if phase != "gas":
            raise ValueError("Use set_mode to select shutdown nozzle phases")
        return super().set_phase(phase)

    def set_mode(self, combusting, phases=()):
        if not isinstance(combusting, bool):
            raise ValueError("combusting must be boolean")
        phases = tuple(sorted(phases))
        if len(phases) != len(set(phases)) or not set(phases) <= {"gas", "liquid"}:
            raise ValueError("Shutdown phases must be unique gas/liquid entries")
        phases = () if combusting else phases
        changed = (getattr(self, "combusting", None), getattr(self, "phases", None)) != (combusting, phases)
        self.combusting, self.phases = combusting, phases
        return changed

    @property
    def equation_names(self):
        return ("flow",) if self.combusting or not self.active else tuple(f"{p}_flow" for p in self.phases)

    @property
    def variable_names(self):
        if self.combusting or not self.active:
            return ("mdot",)
        return tuple(f"mdot_{phase}" for phase in self.phases) + (("gas_area_fraction",) if len(self.phases) == 2 else ())

    def evaluate(self, trial_values, node_states_by_id, source_fluids, directions):
        if self.combusting or not self.active:
            return super().evaluate(trial_values, node_states_by_id, source_fluids, directions)
        self._check_trial_values(trial_values)
        if {f.phase for f in source_fluids.values()} != set(self.phases):
            raise ValueError("Nozzle source phases changed; apply the transition between solves")
        p_from, p_to = self.pressures(node_states_by_id)
        flows, thrust = {}, 0.0
        for phase in self.phases:
            sources = {name: f for name, f in source_fluids.items() if f.phase == phase}
            q = trial_values[f"mdot_{phase}"] / len(sources)
            for name, f in sources.items():
                flows[name] = {"mdot": q, "direction": directions[phase], "fluid": f}
                if phase == "gas":
                    chamber_p = node_states_by_id[self.from_node]["P"]
                    ambient_p = node_states_by_id[self.to_node]["P"]
                    _gas_flux(chamber_p, ambient_p, f)  # Validate thermodynamic inputs.
                    velocity = sqrt(2 * f["gamma"] / (f["gamma"] - 1) * f["R"] * f["T"]
                                    * (1 - (ambient_p / chamber_p) ** ((f["gamma"] - 1) / f["gamma"]))) if chamber_p > ambient_p else 0.0
                else:
                    velocity = _liquid_flux(max(p_from - p_to, 0), f) / f["rho"]
                thrust += q * velocity
        area_fraction = trial_values.get("gas_area_fraction", 1.0 if self.phases == ("gas",) else 0.0)
        properties = {"thrust": thrust}
        if "gas_area_fraction" not in trial_values:
            properties["gas_area_fraction"] = area_fraction
        return BranchState(trial_values=trial_values, flows=flows,
                           dP=p_from - p_to, properties=properties)

    def _residual(self, branch_state, node_states_by_id):
        if self.combusting:
            chamber = node_states_by_id[self.from_node]
            expected = self.throat_area * chamber["P"] / chamber["cstar"] if branch_state.dP > 0 and chamber["cstar"] > 0 else 0.0
            return [(branch_state.mdot - expected) / self.flow_scale]
        p_from = node_states_by_id[self.from_node]["P"]
        p_to = node_states_by_id[self.to_node]["P"]
        fraction = branch_state["gas_area_fraction"]
        areas = {"gas": fraction, "liquid": 1 - fraction}
        equations = []
        for phase in self.phases:
            flows = [flow for flow in branch_state.flows.values() if flow["fluid"].phase == phase]
            q = sum(flow["mdot"] for flow in flows)
            flux = 0.0
            for flow in flows:
                f = flow["fluid"]
                weight = flow["mdot"] / q if q else 1 / len(flows)
                capacity = _gas_flux(p_from, p_to, f) if phase == "gas" else _liquid_flux(max(branch_state.dP, 0), f)
                flux += weight * capacity
            equations.append((q - self.throat_area * areas[phase] * flux) / self.flow_scale)
        return equations


__all__ = ["FluidBranch", "LossComponent", "PumpComponent", "RegulatorComponent",
           "BangBangValveComponent", "NozzleComponent"]
