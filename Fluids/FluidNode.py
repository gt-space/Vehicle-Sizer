"""Continuous node physics for the SUNDIALS network, independent of the old solver.

Nodes own definitions and discrete modes, never accepted solution/history. All
residuals are unscaled SI equations; the assembler owns equation scaling, y/ydot,
constraints, event arming, conservative state mapping and consistent restarts.
"""

from copy import deepcopy
from math import isfinite

import numpy as np

from .FluidState import BranchState, FluidState, NodeState
from .errors import TrialDomainError


Adjacent = list[tuple[float, BranchState]]


def iter_flows(adjacent):
    for incidence, branch in adjacent:
        if branch.enabled:
            for flow in branch.flows.values():
                yield incidence, flow


def net_mdot(adjacent, fluid=None, phase=None):
    return sum(incidence * float(flow["mdot"])
               for incidence, flow in iter_flows(adjacent)
               if (fluid is None or flow["fluid"].fluid == fluid)
               and (phase is None or flow["fluid"].phase == phase))


def fluxes(adjacent, fluid=None):
    mass, energy = 0.0, 0.0
    for incidence, flow in iter_flows(adjacent):
        if fluid is None or flow["fluid"].fluid == fluid:
            q = incidence * float(flow["mdot"])
            mass += q
            energy += q * float(flow["fluid"]["h"])
    return mass, energy


def inventory_residual(*, mass_rate, energy_rate, adjacent, fluid=None,
                       heat_rate=0.0, pressure=0.0, volume_rate=0.0):
    """Return [mass, energy] residuals in kg/s and W; inflows/heat are positive."""
    mdot, hdot = fluxes(adjacent, fluid)
    return np.array([mass_rate - mdot,
                     energy_rate - hdot - heat_rate + pressure * volume_rate])


def _incoming(adjacent):
    # Direction is fixed by the network during an integration segment, rather
    # than selected from a possibly wrong-sign nonlinear trial flow.
    return {flow["fluid"].fluid: flow["fluid"]
            for incidence, flow in iter_flows(adjacent)
            if incidence * flow["direction"] > 0}


class FluidNode:
    """Component definition/mode and local equations, without a stored solution.

    variable_names describes the local y layout; trial_values supplies its
    current numbers. differential_variable_names identifies required ydot keys.
    """
    differential_variable_names = ()
    variable_names = ()
    equation_names = ()

    def __init__(self, node_id, definition, *, fluid_properties=None,
                 combustion_properties=None):
        self.id = node_id
        self.definition = deepcopy(definition)
        self.fluid_properties = fluid_properties
        self.combustion_properties = combustion_properties

    def _check_trial_values(self, trial_values):
        if set(trial_values) != set(self.variable_names):
            raise ValueError(f"Node '{self.id}' requires variables {self.variable_names}")
        if not all(np.isfinite(v) for v in trial_values.values()):
            raise TrialDomainError(f"Node '{self.id}' has non-finite trial values")
        for key in ("P", "T", "T_liq", "T_ull"):
            if key in trial_values and trial_values[key] <= 0:
                raise TrialDomainError(f"Node '{self.id}' requires {key} > 0")

    def _check_trial_derivatives(self, trial_derivatives):
        for name in self.differential_variable_names:
            if name not in trial_derivatives:
                raise ValueError(f"Node '{self.id}' requires derivative of {name}")
            if not np.isfinite(trial_derivatives[name]):
                raise TrialDomainError(f"Node '{self.id}' has non-finite derivative of {name}")

    def initial_values(self):
        """Return configured guesses keyed by variable_names, not a solved state."""
        return {}

    def evaluate(self, trial_values, adjacent=(), node_states_by_id=None, *,
                 axial_specific_force=0.0, boundary_values=None):
        """Build this node's NodeState from trial_values; do not solve equations.

        adjacent holds connected trial BranchStates. node_states_by_id supplies
        dependencies such as ambient pressure, not this node's own output.
        Boundary nodes use boundary_values for externally imposed conditions.
        """
        raise NotImplementedError("Select a concrete node")

    def residual(self, node_state, trial_derivatives, adjacent=(), *, heat_rate=None):
        """Return equation errors using evaluate's output and local ydot values.

        trial_derivatives is keyed by differential variable name (m, U, etc.);
        adjacent supplies trial fluxes to compare with those derivatives.
        """
        raise NotImplementedError("Select a concrete node")

    def outlet(self, node_state, adjacent=(), port=None):
        aliases = {"liquid": "liquid_fluid", "gas": "gas_fluid", "ullage": "gas_fluid",
                   "oxidizer": "oxidizer_fluid", "fuel": "fuel_fluid"}
        selected = self.definition.get(aliases.get(port, ""), port)
        fluids = node_state.fluids
        if selected in fluids:
            return {selected: fluids[selected]}
        if len(fluids) != 1:
            raise ValueError(f"Node '{self.id}' requires an unambiguous outlet port")
        return dict(fluids)

    def event_values(self, node_state):
        return {}

    def _property_limits(self, fluid, pressure, temperature, prefix=""):
        """Positive margins inside the provider's PT domain; no property calls."""
        pressure_bounds, temperature_bounds = self.fluid_properties.state_bounds(fluid)
        events = {}
        for name, value, bounds in (("pressure", pressure, pressure_bounds),
                                    ("temperature", temperature, temperature_bounds)):
            for side, bound, sign in (("low", bounds[0], 1), ("high", bounds[1], -1)):
                if np.isfinite(bound):
                    events[f"{prefix}{name}_{side}"] = sign * (value - bound)
        return events

    def output_state(self, node_state):
        # Internal work coefficients/property objects are not output records.
        return NodeState(trial_values=node_state.trial_values,
                         properties=deepcopy(node_state.properties),
                         fluids=deepcopy(node_state.fluids))


class BoundaryComponent(FluidNode):
    def evaluate(self, trial_values, adjacent=(), node_states_by_id=None, *,
                 axial_specific_force=0.0, boundary_values=None):
        self._check_trial_values(trial_values)
        state = {k: deepcopy(self.definition[k]) for k in ("P", "T", "fluids")
                 if k in self.definition}
        state.update(deepcopy(boundary_values or {}))
        if "P" not in state:
            raise ValueError(f"Boundary '{self.id}' requires a pressure state")
        if not np.isfinite(state["P"]) or state["P"] < 0:
            raise TrialDomainError(f"Boundary '{self.id}' requires finite nonnegative pressure")
        return NodeState.from_dict(state)

    def residual(self, node_state, trial_derivatives, adjacent=(), *, heat_rate=None):
        return np.empty(0)


class JunctionComponent(FluidNode):
    """Zero-volume single-fluid/phase mixer with algebraic P/T balances."""
    variable_names = ("P", "T")
    equation_names = ("mass_rate", "energy_rate")

    def __init__(self, node_id, definition, *, fluid_properties):
        super().__init__(node_id, definition, fluid_properties=fluid_properties)
        self.fluid_name = definition['fluid']
        self.mode = definition['phase']
        if self.mode not in ('liquid', 'gas'):
            raise ValueError('Junction requires a single liquid or gas phase')
        self._check_trial_values(self.initial_values())
        self.reference_T = self.initial_values()['T']

    def select_fluid(self, adjacent):
        """Called only between solves; follow donor replacement after dryout/reversal."""
        incoming = [flow['fluid'] for sign, flow in iter_flows(adjacent)
                    if sign * flow['direction'] > 0]
        identities = {(fluid.fluid, fluid.phase) for fluid in incoming}
        if len(identities) > 1:
            raise ValueError(f"Junction '{self.id}' cannot mix different fluids or phases")
        if identities and identities != {(self.fluid_name, self.mode)}:
            self.fluid_name, self.mode = identities.pop()
            if self.mode not in ('liquid', 'gas'):
                raise ValueError('Junction requires a single liquid or gas phase')
            self.reference_T = float(incoming[0]['T'])
            return self.reference_T
        return None

    def initial_values(self):
        return {name: float(self.definition['state0'][name]) for name in self.variable_names}

    def evaluate(self, trial_values, adjacent=(), node_states_by_id=None, *,
                 axial_specific_force=0.0, boundary_values=None):
        self._check_trial_values(trial_values)
        incoming = {(flow['fluid'].fluid, flow['fluid'].phase)
                    for sign, flow in iter_flows(adjacent) if sign * flow['direction'] > 0}
        if incoming - {(self.fluid_name, self.mode)}:
            raise ValueError(f"Junction '{self.id}' donor changed; select modes between solves")
        p, t = trial_values['P'], trial_values['T']
        pb, tb = self.fluid_properties.state_bounds(self.fluid_name)
        if not pb[0] <= p <= pb[1] or not tb[0] <= t <= tb[1]:
            raise TrialDomainError(f"Junction '{self.id}' outside {self.fluid_name} PT domain")
        properties = self.fluid_properties.state_pt(self.fluid_name, p, t).as_dict()
        return NodeState(trial_values=trial_values,
                         fluids={self.fluid_name: FluidState(self.fluid_name, self.mode, properties)})

    def residual(self, node_state, trial_derivatives, adjacent=(), *, heat_rate=None):
        mass, energy = fluxes(adjacent)
        heat = sum((heat_rate or {}).values())
        if not any(branch.enabled for _, branch in adjacent):
            mass = (node_state['P'] - self.initial_values()['P']) / self.initial_values()['P']
        if not any(sign * flow['direction'] > 0 and flow['mdot'] != 0
                   for sign, flow in iter_flows(adjacent)):
            if heat:
                raise ValueError('A zero-volume junction cannot store heat without throughflow')
            # With no throughput the energy balance is identically zero.
            # Fix the otherwise undetermined temperature; this stores no energy.
            energy = node_state.fluids[self.fluid_name]['cp'] * (node_state['T'] - self.reference_T)
        return np.array([mass, energy + heat])


class VolumeComponent(FluidNode):
    """Rigid pure-fluid storage; saturated mode retains the original vapor outlet."""

    differential_variable_names = ("m", "U")

    @property
    def equation_names(self):
        return ("mass_rate", "energy_rate",
                "volume_closure" if self.mode == "saturated" else "mass_closure", "energy_closure")

    def pressure_rate(self, node_state, trial_derivatives):
        """Differentiate the EOS closures for an ideal regulator's rate constraint."""
        if self.mode == "saturated":
            raise ValueError("Ideal regulation of a saturated volume is not supported")
        f = node_state.fluids[self.fluid_name]
        d = self.fluid_properties.derivatives_pt(self.fluid_name, node_state["P"], node_state["T"])
        m = node_state["m"]
        matrix = np.array([[self.volume * d["drho_dP"], self.volume * d["drho_dT"]],
                           [m * d["du_dP"], m * d["du_dT"]]])
        rhs = [trial_derivatives["m"], trial_derivatives["U"] - f["u"] * trial_derivatives["m"]]
        try:
            return float(np.linalg.solve(matrix, rhs)[0])
        except np.linalg.LinAlgError as error:
            raise TrialDomainError("Singular pressure-rate closure") from error

    def __init__(self, node_id, definition, *, fluid_properties, phase="gas"):
        super().__init__(node_id, definition, fluid_properties=fluid_properties)
        self.volume = float(self.definition["geometry"].volume)
        if not np.isfinite(self.volume) or self.volume <= 0:
            raise ValueError("Tank volume must be finite and positive")
        self.set_mode(phase)

    @property
    def fluid_name(self):
        return self.definition["fluid"]

    @property
    def variable_names(self):
        return ("m", "U", "P", "quality" if self.mode == "saturated" else "T")

    def set_mode(self, mode):
        if mode not in {"gas", "liquid", "saturated"}:
            raise ValueError(f"Unsupported volume mode: {mode}")
        changed = getattr(self, "mode", None) != mode
        self.mode = mode
        return changed

    def initial_values(self):
        # Configured inventories remain fixed inputs to network initialization;
        # inconsistent P/T guesses must not silently replace them.
        state = self.definition["state0"]
        return {name: float(state[name]) for name in self.variable_names}

    def evaluate(self, trial_values, adjacent=(), node_states_by_id=None, *,
                 axial_specific_force=0.0, boundary_values=None):
        self._check_trial_values(trial_values)
        if trial_values["m"] <= 0:
            raise TrialDomainError("A storage volume requires positive mass")
        m, pressure = trial_values["m"], trial_values["P"]
        node_properties = {"mode": self.mode}
        evaluation_data = {}
        if self.mode == "saturated":
            sat = self.fluid_properties.saturation_at_p(self.fluid_name, pressure)
            quality = trial_values["quality"]
            # No clipping: finite continuation across quality=0/1 permits root
            # bracketing. The network must enforce accepted mode inequalities.
            occupied = m * ((1 - quality) / sat.liquid.rho + quality / sat.vapor.rho)
            expected_energy = m * ((1 - quality) * sat.liquid.u + quality * sat.vapor.u)
            properties = sat.vapor.as_dict()
            properties["V"] = quality * m / sat.vapor.rho
            node_properties.update(T=sat.T, occupied_volume=occupied)
            evaluation_data.update(saturation=sat,
                closures=(occupied - self.volume, trial_values["U"] - expected_energy))
            phase = "gas"
        else:
            props = self.fluid_properties.state_pt(self.fluid_name, pressure, trial_values["T"])
            if not np.isfinite(props.rho) or props.rho <= 0:
                raise TrialDomainError("Volume properties require finite positive density")
            properties = props.as_dict()
            properties["V"] = self.volume
            node_properties.update(quality=None, occupied_volume=self.volume)
            evaluation_data["closures"] = (m - props.rho * self.volume, trial_values["U"] - m * props.u)
            phase = self.mode
        return NodeState(trial_values=trial_values, properties=node_properties,
                         evaluation_data=evaluation_data,
                         fluids={self.fluid_name: FluidState(self.fluid_name, phase, properties)})

    def residual(self, node_state, trial_derivatives, adjacent=(), *, heat_rate=None):
        self._check_trial_derivatives(trial_derivatives)
        phase = node_state.fluids[self.fluid_name].phase
        balances = inventory_residual(mass_rate=trial_derivatives["m"], energy_rate=trial_derivatives["U"],
                                      adjacent=adjacent, heat_rate=(heat_rate or {}).get(phase, 0.0))
        return np.r_[balances, node_state.evaluation_data["closures"]]

    def event_values(self, node_state):
        if self.mode == "saturated":
            lower, upper = self.fluid_properties.saturation_bounds(self.fluid_name)
            return {"evaporate": 1.0 - node_state["quality"], "liquid_limit": node_state["quality"],
                    "pressure_low": node_state["P"] - lower, "pressure_high": upper - node_state["P"]}
        events = self._property_limits(self.fluid_name, node_state["P"], node_state["T"])
        if self.mode != "gas" or not self.fluid_properties.supports_saturation(self.fluid_name):
            return events
        lower, upper = self.fluid_properties.saturation_bounds(self.fluid_name)
        if not lower <= node_state["P"] <= upper:
            return events
        sat = self.fluid_properties.saturation_at_p(self.fluid_name, node_state["P"])
        return {**events, "condense": node_state["T"] - sat.T}

    def output_state(self, node_state):
        output = super().output_state(node_state)
        geometry = self.definition["geometry"]
        output.properties.update(mass=node_state["m"], tank_id=self.definition.get("tank_id"),
                            axial_mass=geometry.axial_mass(node_state["m"]))
        if self.mode == "saturated":
            sat = node_state.evaluation_data["saturation"]
            output.properties["phase_states"] = {"liquid": sat.liquid.as_dict(), "gas": sat.vapor.as_dict()}
            if hasattr(geometry, "fill_state"):
                output.properties.update(geometry.fill_state((1 - node_state["quality"]) * node_state["m"] / sat.liquid.rho))
        else:
            output.properties["phase_states"] = {self.mode: node_state.fluids[self.fluid_name].as_dict()}
        return output


class PropellantTankComponent(VolumeComponent):
    """Liquid and pressurant inventories at shared pressure and separate temperatures."""

    def __init__(self, node_id, definition, *, fluid_properties):
        super().__init__(node_id, definition, fluid_properties=fluid_properties)
        m0 = float(self.definition["state0"]["m_liq"])
        if m0 < 0 or not np.isfinite(m0):
            raise ValueError("Initial liquid mass must be finite and nonnegative")
        self.dry_mass = max(min(m0 * 1e-5, 1e-5), 1e-12)
        self.set_mode("two_phase" if m0 > 0 else "gas")

    @property
    def fluid_name(self):
        return self.definition["gas_fluid"]

    @property
    def equation_names(self):
        if self.mode == "two_phase":
            return ("liquid_mass_rate", "liquid_energy_rate", "ullage_mass_rate",
                    "ullage_energy_rate", "liquid_energy_closure", "ullage_energy_closure", "volume_closure")
        return super().equation_names

    def pressure_rate(self, node_state, trial_derivatives):
        if self.mode != "two_phase":
            return super().pressure_rate(node_state, trial_derivatives)
        return float(node_state.evaluation_data["pressure_gradient"]
                     @ np.array([trial_derivatives[k] for k in self.differential_variable_names]))

    @property
    def differential_variable_names(self):
        return ("m_liq", "m_ull", "U_liq", "U_ull") if self.mode == "two_phase" else ("m", "U")

    @property
    def variable_names(self):
        if self.mode == "two_phase":
            return self.differential_variable_names + ("P", "T_liq", "T_ull")
        return super().variable_names

    def set_mode(self, mode):
        if mode == "two_phase":
            changed = getattr(self, "mode", None) != mode
            self.mode = mode
            return changed
        if mode not in {"gas", "saturated"}:
            raise ValueError(f"Unsupported propellant tank mode: {mode}")
        return super().set_mode(mode)

    def initial_values(self):
        state = self.definition["state0"]
        if self.mode == "two_phase":
            return {"P": float(state["P"]), "T_liq": float(state["T"]),
                    "T_ull": float(state["gas_T"]),
                    **{name: float(state[name]) for name in self.differential_variable_names}}
        if float(state["m_liq"]) != 0:
            raise ValueError("Post-dryout inventories must be mapped by the network, not reinitialized from config")
        return {"P": float(state["P"]), "T": float(state["gas_T"]),
                "m": float(state["m_ull"]), "U": float(state["U_ull"])}

    @staticmethod
    def _thermodynamic_response(m_l, m_g, liquid, gas, dl, dg):
        """Return liquid-volume and pressure gradients along the EOS closures.

        Eliminate T_liq/T_ull from dU_i=d(m_i*u_i), then impose dV_l+dV_g=0.
        No inner matrix solve or algebraic ydot is needed. Invalid elimination
        is a recoverable trial-domain error, never a fallback to another solve.
        """
        inputs = (m_l, m_g, liquid.rho, gas.rho, liquid.u, gas.u,
                  dl['drho_dP'], dl['drho_dT'], dl['du_dP'], dl['du_dT'],
                  dg['drho_dP'], dg['drho_dT'], dg['du_dP'], dg['du_dT'])
        if (not all(map(isfinite, inputs)) or min(m_l, m_g, liquid.rho, gas.rho) <= 0
                or dl['du_dT'] == 0 or dg['du_dT'] == 0):
            raise TrialDomainError('Invalid liquid/ullage thermodynamic response inputs')
        # Cancel phase mass before division: small inventories need no cutoff.
        fb = -dl['drho_dT'] / liquid.rho**2 / dl['du_dT']
        gd = -dg['drho_dT'] / gas.rho**2 / dg['du_dT']
        lp, gp = -m_l * dl['drho_dP'] / liquid.rho**2, -m_g * dg['drho_dP'] / gas.rho**2
        lt, gt = fb * m_l * dl['du_dP'], gd * m_g * dg['du_dP']
        kl, kg = lp - lt, gp - gt
        denominator = kl + kg
        scale = max(abs(lp), abs(gp), abs(lt), abs(gt))
        if (not all(map(isfinite, (fb, gd, denominator, scale))) or scale == 0
                or abs(denominator) <= 1e-10 * scale):
            raise TrialDomainError(
                f'Singular liquid/ullage thermodynamic response: denominator={denominator:g}, scale={scale:g}')
        ql, qg = 1 / liquid.rho - fb * liquid.u, 1 / gas.rho - gd * gas.u
        pressure = np.array([-ql, -qg, -fb, -gd]) / denominator
        # Use kg/(kl+kg), instead of 1-kl/(kl+kg), to avoid cancellation.
        gradient = np.array([ql * (kg / denominator), -qg * (kl / denominator),
                             fb * (kg / denominator), -gd * (kl / denominator)])
        if not np.isfinite(gradient).all() or not np.isfinite(pressure).all():
            raise TrialDomainError('Non-finite liquid/ullage thermodynamic response')
        return gradient, pressure

    @staticmethod
    def _volume_gradient(m_l, m_g, liquid, gas, dl, dg):
        return PropellantTankComponent._thermodynamic_response(m_l, m_g, liquid, gas, dl, dg)[0]

    def evaluate(self, trial_values, adjacent=(), node_states_by_id=None, *,
                 axial_specific_force=0.0, boundary_values=None):
        if self.mode != "two_phase":
            return super().evaluate(trial_values, adjacent, node_states_by_id,
                                    axial_specific_force=axial_specific_force, boundary_values=boundary_values)
        self._check_trial_values(trial_values)
        m_l, m_g, pressure = trial_values["m_liq"], trial_values["m_ull"], trial_values["P"]
        if m_l <= 0 or m_g <= 0:
            raise TrialDomainError("Wet tank requires positive liquid and ullage inventories")
        liquid_name, gas_name = self.definition["liquid_fluid"], self.definition["gas_fluid"]
        liquid = self.fluid_properties.state_pt(liquid_name, pressure, trial_values["T_liq"])
        gas = self.fluid_properties.state_pt(gas_name, pressure, trial_values["T_ull"])
        if any(not np.isfinite(f.rho) or f.rho <= 0 for f in (liquid, gas)):
            raise TrialDomainError("Tank properties require finite positive densities")
        dl = self.fluid_properties.derivatives_pt(liquid_name, pressure, trial_values["T_liq"])
        dg = self.fluid_properties.derivatives_pt(gas_name, pressure, trial_values["T_ull"])
        vl, vg = m_l / liquid.rho, m_g / gas.rho
        fill = self.definition["geometry"].fill_state(vl)
        outlet_p = pressure + liquid.rho * axial_specific_force * fill["fill_height"]
        gradient, pressure_gradient = self._thermodynamic_response(m_l, m_g, liquid, gas, dl, dg)
        properties = {"mode": self.mode,
                      "port_pressure": {"liquid": outlet_p, liquid_name: outlet_p,
                                        "ullage": pressure, "gas": pressure, gas_name: pressure}}
        evaluation_data = {"closures": (trial_values["U_liq"] - m_l * liquid.u,
                                       trial_values["U_ull"] - m_g * gas.u, vl + vg - self.volume),
                           "volume_gradient": gradient, "pressure_gradient": pressure_gradient}
        return NodeState(trial_values=trial_values, properties=properties,
                         evaluation_data=evaluation_data, fluids={
            liquid_name: FluidState(liquid_name, "liquid", {**liquid.as_dict(), "V": vl,
                                   "contact_area": fill["liquid_contact_area"]}),
            gas_name: FluidState(gas_name, "gas", {**gas.as_dict(), "V": vg,
                                "contact_area": fill["ullage_contact_area"]})})

    def residual(self, node_state, trial_derivatives, adjacent=(), *, heat_rate=None):
        if self.mode != "two_phase":
            return super().residual(node_state, trial_derivatives, adjacent, heat_rate=heat_rate)
        self._check_trial_derivatives(trial_derivatives)
        dv = float(node_state.evaluation_data["volume_gradient"]
                   @ np.array([trial_derivatives[k] for k in self.differential_variable_names]))
        heat = heat_rate or {}
        liquid = inventory_residual(mass_rate=trial_derivatives["m_liq"], energy_rate=trial_derivatives["U_liq"],
                                    adjacent=adjacent, fluid=self.definition["liquid_fluid"],
                                    heat_rate=heat.get("liquid", 0.0), pressure=node_state["P"], volume_rate=dv)
        gas = inventory_residual(mass_rate=trial_derivatives["m_ull"], energy_rate=trial_derivatives["U_ull"],
                                 adjacent=adjacent, fluid=self.definition["gas_fluid"],
                                 heat_rate=heat.get("gas", 0.0), pressure=node_state["P"], volume_rate=-dv)
        return np.r_[liquid, gas, node_state.evaluation_data["closures"]]

    def event_values(self, node_state):
        if self.mode == "two_phase":
            return {"dryout": node_state["m_liq"] - self.dry_mass,
                    **self._property_limits(self.definition["liquid_fluid"], node_state["P"], node_state["T_liq"], "liquid_"),
                    **self._property_limits(self.definition["gas_fluid"], node_state["P"], node_state["T_ull"], "ullage_")}
        return super().event_values(node_state)

    def output_state(self, node_state):
        # Propellant geometry takes phase masses, unlike pure-volume geometry.
        output = FluidNode.output_state(self, node_state)
        geometry = self.definition["geometry"]
        if self.mode == "two_phase":
            vl = node_state.fluids[self.definition["liquid_fluid"]]["V"]
            ml, mg = node_state["m_liq"], node_state["m_ull"]
            phases = {f.phase: f.as_dict() for f in node_state.fluids.values()}
        elif self.mode == "saturated":
            sat = node_state.evaluation_data["saturation"]
            mg = node_state["quality"] * node_state["m"]
            ml = (1 - node_state["quality"]) * node_state["m"]
            vl = ml / sat.liquid.rho
            phases = {"liquid": sat.liquid.as_dict(), "gas": sat.vapor.as_dict()}
        else:
            vl, ml, mg = 0.0, 0.0, node_state["m"]
            phases = {"gas": node_state.fluids[self.fluid_name].as_dict()}
        output.properties.update(mass=ml + mg, tank_id=self.definition.get("tank_id"),
                            phase_states=phases, **geometry.fill_state(vl),
                            axial_mass=geometry.axial_mass(liquid_volume=vl, liquid_mass=ml, ullage_mass=mg))
        return output


class CombustorComponent(FluidNode):
    """Steady combustion, or independent gas/liquid continuity after shutdown."""

    def __init__(self, node_id, definition, *, combustion_properties):
        super().__init__(node_id, definition, combustion_properties=combustion_properties)
        self.set_mode("combusting")

    @property
    def equation_names(self):
        return ("mass_rate",) if self.mode == "combusting" else tuple(f"{p}_mass_rate" for p in self.phases)

    @property
    def variable_names(self):
        return ("P",) if self.mode == "combusting" or self.phases else ()

    def set_mode(self, mode, phases=(), *, reason=None):
        phases = tuple(sorted(phases))
        if mode not in {"combusting", "shutdown"}:
            raise ValueError(f"Unsupported chamber mode: {mode}")
        if len(phases) != len(set(phases)) or not set(phases) <= {"gas", "liquid"}:
            raise ValueError("Shutdown phases must be unique liquid/gas entries")
        phases = () if mode == "combusting" else phases
        changed = (getattr(self, "mode", None), getattr(self, "phases", None)) != (mode, phases)
        self.mode, self.phases = mode, phases
        self.shutdown_reason = reason if mode == "shutdown" else None
        return changed

    def initial_values(self):
        return {"P": float(self.definition["P0"])} if self.variable_names else {}

    def evaluate(self, trial_values, adjacent=(), node_states_by_id=None, *,
                 axial_specific_force=0.0, boundary_values=None, reference_state=None):
        self._check_trial_values(trial_values)
        definition = self.definition
        ambient = node_states_by_id[definition["ambient_node"]]["P"]
        if self.mode == "shutdown":
            sources = _incoming(adjacent)
            if {f.phase for f in sources.values()} != set(self.phases):
                raise ValueError("Chamber source phases changed; apply transition between solves")
            properties = {"cstar": 0.0, "Cf": 0.0, "MR": 0.0}
            if "P" not in trial_values:
                properties["P"] = ambient
            return NodeState(trial_values=trial_values, properties=properties, fluids=deepcopy(sources))
        ox = net_mdot(adjacent, fluid=definition["oxidizer_fluid"])
        fuel = net_mdot(adjacent, fluid=definition["fuel_fluid"])
        flowing = ox > 0 and fuel > 0
        mr = float(np.clip(ox / fuel, 0.1, 10.0)) if flowing else 0.0
        if flowing:
            product = self.combustion_properties.evaluate(
                chamber_pressure=trial_values["P"], mixture_ratio=mr, ambient_pressure=ambient,
                expansion_ratio=definition["expansion_ratio"],
                cstar_efficiency=definition["cstar_efficiency"],
                cf_efficiency=definition["cf_efficiency"]).as_dict()
        else:
            product = {name: 0.0 for name in ("cstar", "Cf", "R", "gamma", "T")}
        name = definition["combustion_fluid"]
        # Retain the original no-reactant trial transport rule, but take the
        # reference explicitly from the network instead of hidden accepted state.
        prior = reference_state.fluids.get(name) if reference_state is not None else None
        transport = product if flowing else (prior or {"R": 300.0, "gamma": 1.2, "T": 300.0})
        properties = {k: transport[k] for k in ("R", "gamma", "T")}
        properties["h"] = transport["gamma"] * transport["R"] * transport["T"] / (transport["gamma"] - 1)
        node_properties = {"cstar": product["cstar"], "Cf": product["Cf"], "MR": mr,
                           "mdot_oxidizer": ox, "mdot_fuel": fuel}
        evaluation_data = {"reactants": {
            "oxidizer_unavailable": net_mdot(adjacent, fluid=definition["oxidizer_fluid"], phase="liquid"),
            "fuel_unavailable": net_mdot(adjacent, fluid=definition["fuel_fluid"], phase="liquid")}}
        return NodeState(trial_values=trial_values, properties=node_properties,
                         evaluation_data=evaluation_data, fluids={name: FluidState(name, "gas", properties)})

    def residual(self, node_state, trial_derivatives, adjacent=(), *, heat_rate=None):
        if self.mode == "combusting":
            return np.array([net_mdot(adjacent)])
        return np.array([net_mdot(adjacent, phase=phase) for phase in self.phases])

    def outlet(self, node_state, adjacent=(), port=None):
        if self.mode == "combusting":
            return super().outlet(node_state, adjacent, port)
        selected = self.definition.get({"oxidizer": "oxidizer_fluid", "fuel": "fuel_fluid"}.get(port, ""), port)
        if selected in node_state.fluids:
            return {selected: node_state.fluids[selected]}
        # The nozzle's configured product port now carries all nonreacting
        # streams. It must not require renaming the branch after shutdown.
        return dict(node_state.fluids)

    def event_values(self, node_state):
        return dict(node_state.evaluation_data["reactants"]) if self.mode == "combusting" else {}

    def output_state(self, node_state):
        output = super().output_state(node_state)
        output.properties.update(mode=self.mode, shutdown_reason=self.shutdown_reason)
        return output


__all__ = ["FluidNode", "BoundaryComponent", "JunctionComponent", "VolumeComponent",
           "PropellantTankComponent", "CombustorComponent", "TrialDomainError",
           "inventory_residual", "net_mdot"]
