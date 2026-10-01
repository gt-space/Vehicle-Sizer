"""Component assembly and event-driven IDAS integration.

Physics lives in components. This class owns topology, trial routing, numerical
layouts, discrete transitions and accepted snapshots.
"""
from collections import Counter
from copy import copy, deepcopy
from dataclasses import dataclass, field

import numpy as np

from .FluidNode import (BoundaryComponent, JunctionComponent, VolumeComponent,
                        PropellantTankComponent, CombustorComponent)
from .FluidBranch import (LossComponent, PumpComponent, RegulatorComponent,
                          BangBangValveComponent, NozzleComponent)
from .FluidState import NodeState
from errors import TrialDomainError, ResidualAcceptanceError, SolverConvergenceError
from .Sundials.ida_session import IdaSession, IdaStep
from .Sundials.jacobian import install as install_jacobian
from Thermals.heat_sources import build_heat_source, evaluate_heat, NoHeating, Aeroheating
from .helpers.pressure_tracking import PressureTracking


@dataclass
class NetworkState:
    nodes: dict = field(default_factory=dict)
    branches: dict = field(default_factory=dict)

    @property
    def node(self):
        return self.nodes


class FluidNetwork:
    def __init__(self, nodes, branches, *, fluid_properties,
                 combustion_properties=None, tolerances=None, constraint_monitor=None,
                 stop_at_shutdown=False, stop_at_triple_point=False):
        if not isinstance(stop_at_shutdown, bool) or not isinstance(stop_at_triple_point, bool):
            raise ValueError('Stopping policies must be boolean')
        self.stop_at_shutdown, self.stop_at_triple_point = stop_at_shutdown, stop_at_triple_point
        self.frozen = False
        self.stop_reason = None
        self._triple_limits = {}
        self.node_definitions, self.branch_definitions = deepcopy(nodes), deepcopy(branches)
        self.fluid_properties, self.combustion_properties = fluid_properties, combustion_properties
        kinds = {self._kind(b) for b in branches.values()}
        # Infer control from the selected wiring, including custom/mixed templates.
        default_rtol = 1e-5 if 'bang_bang_valve' in kinds else 1e-4 if 'regulator' in kinds else 1e-7
        self.options = dict(rtol=default_rtol, residual_tolerance=1e-5, event_tolerance=1e-8, event_time_tolerance=1e-8,
                            regulator_pressure_tolerance=0.0, residual_retries=2,
                            retry_rtol_factor=0.1, retry_rtol_floor=1e-8,
                            max_events=1000, initialization_horizon=0.001, max_step=np.inf,
                            suppress_algebraic_error=False, jacobian='colored', verify_jacobian=False)
        self.options.update(tolerances or {})
        allowed = {'rtol', 'residual_tolerance', 'event_tolerance', 'event_time_tolerance',
                   'regulator_pressure_tolerance', 'residual_retries', 'retry_rtol_factor', 'retry_rtol_floor',
                   'max_events', 'initialization_horizon', 'max_step', 'suppress_algebraic_error',
                   'atol', 'equation_scales', 'jacobian', 'verify_jacobian'}
        if set(self.options) - allowed:
            raise ValueError(f'Unknown solver options: {sorted(set(self.options) - allowed)}')
        if not isinstance(self.options['suppress_algebraic_error'], bool):
            raise ValueError('suppress_algebraic_error must be boolean')
        if self.options['jacobian'] not in ('colored', 'dense'):
            raise ValueError("jacobian must be 'colored' or 'dense'")
        if not isinstance(self.options['verify_jacobian'], bool):
            raise ValueError('verify_jacobian must be boolean')
        for key in ('rtol', 'residual_tolerance', 'event_tolerance', 'event_time_tolerance', 'initialization_horizon', 'max_events'):
            if not np.isfinite(self.options[key]) or self.options[key] <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if self.options['max_step'] <= 0 or np.isnan(self.options['max_step']):
            raise ValueError('max_step must be positive')
        if (not np.isfinite(self.options['regulator_pressure_tolerance'])
                or self.options['regulator_pressure_tolerance'] < 0):
            raise ValueError('regulator_pressure_tolerance must be nonnegative and finite')
        retries = self.options['residual_retries']
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError('residual_retries must be a nonnegative integer')
        if not 0 < self.options['retry_rtol_factor'] < 1:
            raise ValueError('retry_rtol_factor must be between zero and one')
        if not np.isfinite(self.options['retry_rtol_floor']) or self.options['retry_rtol_floor'] <= 0:
            raise ValueError('retry_rtol_floor must be positive and finite')
        # Accuracy used by IDAS can tighten without changing physical event guards.
        self.effective_rtol = self.options['rtol']
        self.effective_max_step = self.options['max_step']
        self.retry_diagnostics = []
        self.nodes = {k: self._make_node(k, v) for k, v in nodes.items()}
        self.heat_sources = {key: build_heat_source(definition.get('thermal'), definition.get('geometry'))
                             for key, definition in nodes.items()}
        for key, source in self.heat_sources.items():
            if not isinstance(source, NoHeating) and not isinstance(self.nodes[key], (VolumeComponent, JunctionComponent)):
                raise ValueError(f'Node {key!r} does not support prescribed energy input')
        self.branches = {k: self._make_branch(k, v) for k, v in branches.items()}
        if set(self.nodes) & set(self.branches):
            raise ValueError('Node and branch IDs must be distinct')
        self.connections = {k: [] for k in self.nodes}
        for k, branch in self.branches.items():
            for node, sign in ((branch.from_node, -1), (branch.to_node, 1)):
                if node not in self.nodes:
                    raise ValueError(f"Branch '{k}' references unknown node '{node}'")
                self.connections[node].append((sign, k))
        self.regulator_modes = {}
        regulated = set()
        for k, branch in self.branches.items():
            if isinstance(branch, RegulatorComponent) and isinstance(self.nodes[branch.to_node], VolumeComponent):
                if branch.to_node in regulated:
                    raise ValueError('Multiple ideal regulators on one storage volume are underdetermined')
                regulated.add(branch.to_node)
                self.regulator_modes[k] = 'capacity'
        self.directions = {k: int(v.get('direction', 1)) for k, v in branches.items()}
        if any(d not in (-1, 1) for d in self.directions.values()):
            raise ValueError('Branch directions must be -1 or 1')
        self.constraint_monitor = constraint_monitor
        self.state, self.time = NetworkState(), 0.0
        self.pressure_tracking = PressureTracking(nodes)
        self.events, self.remainders = [], []
        self.session = None
        self.y = self.ydot = None
        self.bcs, self.heat_rate, self.axial_specific_force = {}, {}, 0.0
        self.constraints, self.constraint_times = {}, {}
        self.last_diagnostic = None

    @staticmethod
    def _kind(definition):
        keys = [k for k in ('component', 'model') if k in definition]
        if len(keys) != 1:
            raise ValueError('A definition requires exactly one of component or model')
        return definition[keys[0]]

    def _make_node(self, key, definition):
        kind = self._kind(definition)
        if kind == 'boundary':
            return BoundaryComponent(key, definition)
        if kind == 'junction':
            return JunctionComponent(key, definition, fluid_properties=self.fluid_properties)
        if kind in ('volume', 'pressurant_tank'):
            return VolumeComponent(key, definition, fluid_properties=self.fluid_properties,
                                   phase=definition.get('phase', 'gas'))
        if kind == 'propellant_tank':
            return PropellantTankComponent(key, definition, fluid_properties=self.fluid_properties)
        if kind == 'combustor':
            return CombustorComponent(key, definition, combustion_properties=self.combustion_properties)
        raise ValueError(f"Unsupported node kind '{kind}' for '{key}'")

    def _make_branch(self, key, definition):
        kind = self._kind(definition)
        classes = {'loss': LossComponent, 'incompressible_loss': LossComponent,
                   'compressible_loss': LossComponent, 'pump': PumpComponent,
                   'regulator': RegulatorComponent, 'bang_bang_valve': BangBangValveComponent,
                   'nozzle': NozzleComponent}
        if kind not in classes:
            raise ValueError(f"Unsupported branch kind '{kind}' for '{key}'")
        kwargs = {'phase': definition.get('phase', 'gas' if kind == 'compressible_loss' else 'liquid')} if kind in ('loss', 'incompressible_loss', 'compressible_loss', 'pump') else {}
        branch = classes[kind](key, definition, **kwargs)
        branch.set_enabled(definition.get('enabled', True))
        return branch

    def _build_layout(self, guesses):
        self.variable_index, self.equation_index = {}, {}
        self.variable_slices, self.equation_slices = {}, {}
        values, ids, atols, scales = [], [], [], []
        defaults = {'P': 0.001, 'T': 1e-7, 'T_liq': 1e-7, 'T_ull': 1e-7,
                    'm': 1e-11, 'm_liq': 1e-11, 'm_ull': 1e-11,
                    'U': 1e-6, 'U_liq': 1e-6, 'U_ull': 1e-6}
        for key, component in {**self.nodes, **self.branches}.items():
            start = len(values)
            for name in component.variable_names:
                qualified = f'{key}.{name}'
                self.variable_index[qualified] = len(values)
                values.append(float(guesses[key][name]))
                ids.append(name in component.differential_variable_names)
                # Absolute flow accuracy of 1 mg/s permits the square-root law
                # to cross zero without demanding a differentiable mdot(t).
                default = 1e-6 if name.startswith('mdot') else defaults.get(name, 1e-10)
                atols.append(self.options.get('atol', {}).get(qualified, default))
            self.variable_slices[key] = slice(start, len(values))
            start = len(scales)
            for name in component.equation_names:
                qualified = f'{key}.{name}'
                self.equation_index[qualified] = len(scales)
                scale = 1.0
                if key in self.nodes:
                    if 'energy' in name:
                        field_name = 'U_liq' if name.startswith('liquid') else 'U_ull' if name.startswith('ullage') else 'U'
                        scale = max(abs(guesses[key].get(field_name, 1.)), 1.)
                        if isinstance(component, JunctionComponent):
                            scale = 1e6  # W; independently configurable per junction
                    elif name == 'volume_closure':
                        scale = component.volume
                    elif name == 'mass_closure':
                        scale = max(abs(guesses[key]['m']), 1e-6)
                scales.append(self.options.get('equation_scales', {}).get(qualified, scale))
            self.equation_slices[key] = slice(start, len(scales))
        if not values or len(values) != len(scales):
            raise ValueError(f'Network has {len(values)} unknowns and {len(scales)} equations')
        self.differential = np.asarray(ids, dtype=int)
        self.atol, self.row_scales = np.asarray(atols), np.asarray(scales)
        if not np.isfinite(self.row_scales).all() or np.any(self.row_scales <= 0):
            raise ValueError('Equation scales must be positive and finite')
        self._trial_variables = tuple((key, tuple(component.variable_names), self.variable_slices[key])
                                      for key, component in {**self.nodes, **self.branches}.items())
        self._trial_derivatives = tuple((key, tuple((name, self.variable_index[f'{key}.{name}'])
                                        for name in node.differential_variable_names))
                                       for key, node in self.nodes.items())
        self._transport_plan = None
        return np.asarray(values)

    def _unpack(self, y):
        return {key: dict(zip(names, y[indices])) for key, names, indices in self._trial_variables}

    def _adjacent(self, key, branches):
        return [(sign, branches[bid]) for sign, bid in self.connections[key] if bid in branches]

    def _boundary(self, key, t):
        value = self.bcs.get(key, {})
        return value(t) if callable(value) else value

    def _check_property_domain(self, key, node, values):
        if node.mode == 'two_phase':
            fluids = [(node.definition['liquid_fluid'], 'T_liq'), (node.definition['gas_fluid'], 'T_ull')]
        elif node.mode == 'saturated':
            lo, hi = node.fluid_properties.saturation_bounds(node.fluid_name)
            if not lo <= values['P'] <= hi:
                raise TrialDomainError(f'{key}.P outside saturation domain')
            return
        else:
            fluids = [(node.fluid_name, 'T')]
        for fluid, temperature in fluids:
            pb, tb = node.fluid_properties.state_bounds(fluid)
            if not pb[0] <= values['P'] <= pb[1] or not tb[0] <= values[temperature] <= tb[1]:
                raise TrialDomainError(f'{key}: trial outside {fluid} PT domain')

    def _transport_operations(self):
        """Order property transport once per layout/direction/activation change.

        During mode selection this is a live iterator, so downstream operations
        see phase/shutdown changes made by the preceding evaluation.
        """
        ready = {key for key, node in self.nodes.items()
                 if isinstance(node, (BoundaryComponent, VolumeComponent))}
        pending = set(self.branches)
        while pending or len(ready) < len(self.nodes):
            progress = False
            for key in sorted(pending):
                branch, direction = self.branches[key], self.directions[key]
                donor, port = ((branch.from_node, branch.from_port) if direction > 0
                               else (branch.to_node, branch.to_port))
                if branch.active and donor not in ready:
                    continue
                yield ('branch', key, donor, port, dict(main=direction, gas=direction, liquid=direction))
                pending.remove(key)
                progress = True
            for key in self.nodes:
                if key in ready:
                    continue
                incoming = tuple((sign, bid) for sign, bid in self.connections[key]
                                 if self.branches[bid].active and sign * self.directions[bid] > 0)
                if any(bid in pending for _, bid in incoming):
                    continue
                yield ('node', key, incoming)
                ready.add(key)
                progress = True
            if not progress:
                missing = sorted(set(self.nodes) - ready)
                raise ValueError(f'Unsupported transport cycle or missing donor: nodes={missing}, branches={sorted(pending)}')

    def evaluate_trial(self, t, y, *, select_modes=False):
        """Fresh trial records using cached routing; modes change only outside IDA."""
        if select_modes:
            self._transport_plan = None
            operations = self._transport_operations()
        else:
            signature = tuple((key, branch.active, self.directions[key]) for key, branch in self.branches.items())
            if self._transport_plan is None or self._transport_plan[0] != signature:
                self._transport_plan = (signature, tuple(self._transport_operations()))
            operations = self._transport_plan[1]
        values = self._unpack(y)
        nodes = {key: NodeState(trial_values=values[key]) for key in self.nodes}
        branches = {}
        for key, node in self.nodes.items():
            if isinstance(node, (BoundaryComponent, VolumeComponent)):
                if isinstance(node, VolumeComponent):
                    self._check_property_domain(key, node, values[key])
                nodes[key] = node.evaluate(values[key], axial_specific_force=self.axial_specific_force,
                                           boundary_values=self._boundary(key, t))
            elif not node.variable_names and isinstance(node, CombustorComponent):
                nodes[key].properties['P'] = self._boundary(node.definition['ambient_node'], t).get(
                    'P', self.nodes[node.definition['ambient_node']].definition['P'])
        for operation in operations:
            key = operation[1]
            if operation[0] == 'branch':
                _, _, donor, port, directions = operation
                branch = self.branches[key]
                source = self.nodes[donor].outlet(nodes[donor], self._adjacent(donor, branches), port) if branch.active else {}
                if select_modes and branch.active and not isinstance(branch, NozzleComponent):
                    if len(source) != 1:
                        raise ValueError(f"Branch '{key}' needs one donor fluid")
                    branch.set_phase(next(iter(source.values())).phase)
                    if set(values[key]) != set(branch.variable_names):
                        return None
                branches[key] = branch.evaluate(values[key], nodes, source, directions)
            else:
                node = self.nodes[key]
                adjacent = [(sign, branches[bid]) for sign, bid in operation[2]]
                extra = {}
                if isinstance(node, JunctionComponent) and select_modes:
                    temperature = node.select_fluid(adjacent)
                    if temperature is not None:
                        values[key]['T'] = temperature
                        self.y[self.variable_index[f'{key}.T']] = temperature
                if isinstance(node, CombustorComponent):
                    if select_modes:
                        phases = {flow['fluid'].phase for _, b in adjacent for flow in b.flows.values()}
                        reactants = {flow['fluid'].fluid for _, b in adjacent for flow in b.flows.values()
                                     if flow['fluid'].phase == 'liquid'}
                        if node.mode == 'shutdown' or not {node.definition['oxidizer_fluid'], node.definition['fuel_fluid']} <= reactants:
                            previous = node.mode
                            changed = node.set_mode('shutdown', phases, reason=node.shutdown_reason or 'reactant_unavailable')
                            if changed:
                                self.events.append(dict(time_s=self.time, kind='node', component=key,
                                                        name='shutdown' if previous == 'combusting' else 'phase_change',
                                                        phases=tuple(sorted(phases)), reason=node.shutdown_reason))
                            for nozzle in self.branches.values():
                                if isinstance(nozzle, NozzleComponent) and nozzle.from_node == key:
                                    nozzle.set_mode(False, phases)
                            if (set(values[key]) != set(node.variable_names)
                                    or any(set(values[k]) != set(b.variable_names) for k, b in self.branches.items())):
                                return None
                    extra['reference_state'] = self.state.nodes.get(key)
                nodes[key] = node.evaluate(values[key], adjacent, nodes, **extra)
        adjacent = {key: self._adjacent(key, branches) for key in self.nodes}
        for key, node in self.nodes.items():
            if isinstance(node, VolumeComponent):
                allowed = set(nodes[key].fluids)
                for sign, branch in adjacent[key]:
                    for flow in branch.flows.values():
                        if sign * flow['direction'] > 0 and flow['fluid'].fluid not in allowed:
                            raise ValueError(f"Storage '{key}' cannot mix incoming {flow['fluid'].fluid} with {sorted(allowed)}")
        return nodes, branches, adjacent

    def _derivatives(self, ydot):
        return {key: {name: ydot[index] for name, index in indices}
                for key, indices in self._trial_derivatives}

    def _heat_for(self, key, state, time):
        source = self.heat_sources[key]
        if key in self.heat_rate:
            if not isinstance(source, (NoHeating, Aeroheating)):
                raise ValueError(f'Conflicting heat sources for node {key!r}')
            source = self.heat_rate[key]
        return evaluate_heat(source, state, time)

    def residual(self, t, y, ydot, out):
        try:
            nodes, branches, adjacent = self.evaluate_trial(t, y)
            derivatives = self._derivatives(ydot)
            for key, node in self.nodes.items():
                heat = self._heat_for(key, nodes[key], t)
                out[self.equation_slices[key]] = node.residual(nodes[key], derivatives[key], adjacent[key], heat_rate=heat)
            for key, branch in self.branches.items():
                if key in self.regulator_modes and branch.active:
                    mode, q = self.regulator_modes[key], branches[key].mdot
                    if mode == 'regulating':
                        target = branch.to_node
                        rows = [self.nodes[target].pressure_rate(nodes[target], derivatives[target]) / branch.target]
                    else:
                        expected = branches[key]['max_mdot'] if mode == 'capacity' else 0.
                        rows = [(q - expected) / branch.flow_scale]
                else:
                    rows = branch.residual(branches[key], nodes)
                out[self.equation_slices[key]] = rows
            out[:] /= self.row_scales
        except TrialDomainError as error:
            raise TrialDomainError(f'Network trial at t={t:g}: {error}') from error

    def _event_margins(self, t, y, ydot):
        values = self._unpack(y)
        margins = {}
        # Storage events use raw P/T/inventories, not a full property evaluation.
        for key, node in self.nodes.items():
            if isinstance(node, VolumeComponent):
                for name, value in node.event_values(NodeState(trial_values=values[key])).items():
                    # End the supported model slightly inside hard property limits.
                    # This leaves a valid neighborhood for root interpolation.
                    if name.endswith(('_low', '_high')):
                        variable = 'P' if 'pressure' in name else 'T_liq' if name.startswith('liquid_') else 'T_ull' if name.startswith('ullage_') else 'T'
                        index = self.variable_index.get(f'{key}.{variable}')
                        if index is not None:
                            value -= 10 * (self.atol[index] + self.options['rtol'] * abs(y[index]))
                    margins[('node', key, name)] = value
                if self.stop_at_triple_point:
                    names = ((node.definition['liquid_fluid'], 'T_liq'), (node.definition['gas_fluid'], 'T_ull')) if node.mode == 'two_phase' else ((node.fluid_name, 'T'),)
                    for fluid, temperature in names:
                        if not self.fluid_properties.supports_saturation(fluid):
                            continue
                        if fluid not in self._triple_limits:
                            low, _ = self.fluid_properties.saturation_bounds(fluid)
                            self._triple_limits[fluid] = (low, self.fluid_properties.saturation_at_p(fluid, low).T)
                        low_p, low_t = self._triple_limits[fluid]
                        variable, limit = ('P', low_p) if node.mode == 'saturated' else (temperature, low_t)
                        index = self.variable_index[f'{key}.{variable}']
                        buffer = 10 * (self.atol[index] + self.options['rtol'] * abs(y[index]))
                        margins[('node', key, f'triple_point:{fluid}')] = (values[key][variable] - limit - buffer) / limit
        raw = {key: NodeState(trial_values=values[key]) for key in self.nodes}
        for key, node in self.nodes.items():
            if isinstance(node, BoundaryComponent):
                raw[key] = node.evaluate({}, boundary_values=self._boundary(key, t))
        needs_trial = bool(self.regulator_modes) or any(isinstance(n, CombustorComponent) and n.mode == 'combusting' for n in self.nodes.values())
        evaluated = self.evaluate_trial(t, y) if needs_trial else None
        for key, branch in self.branches.items():
            if isinstance(branch, BangBangValveComponent):
                margins.update({('branch', key, name): v for name, v in branch.event_values(raw).items()})
            elif key in self.regulator_modes and branch.active:
                mode = self.regulator_modes[key]
                if mode == 'regulating':
                    b = evaluated[1][key]
                    margins[('branch', key, 'close_limit')] = b.mdot / branch.flow_scale
                    margins[('branch', key, 'capacity_limit')] = (b['max_mdot'] - b.mdot) / branch.flow_scale
                else:
                    delta = (raw[branch.to_node]['P'] - branch.target) / branch.target
                    margins[('branch', key, 'regulate')] = delta if mode == 'closed' else -delta
            elif branch.active and isinstance(branch, (LossComponent, PumpComponent)) and not isinstance(branch, RegulatorComponent):
                margins[('branch', key, 'reverse')] = self.directions[key] * values[key]['mdot']
        if evaluated:
            for key, node in self.nodes.items():
                if isinstance(node, CombustorComponent):
                    margins.update({('node', key, name): v for name, v in node.event_values(evaluated[0][key]).items()})
        return margins

    def root_values(self, t, y, ydot, out):
        margins = self._event_margins(t, y, ydot)
        for i, event in enumerate(self.root_names):
            # Conditional saturation roots keep their identity throughout a segment.
            out[i] = margins.get(event, 1.)

    def _seed_consistent(self):
        """Bounded starting guess for large equation changes, never a time step.

        Only algebraic y and differential ydot are adjusted. IDAS still performs
        the final consistency solve. This avoids extrapolating combustion guesses
        through the much lower passive-discharge pressure after shutdown.
        """
        from scipy.optimize import least_squares

        algebraic = np.flatnonzero(1 - self.differential)
        differential = np.flatnonzero(self.differential)
        names = list(self.variable_index)
        initial = np.r_[self.y[algebraic], self.ydot[differential]]
        scale = np.maximum(np.abs(initial), 1.)
        lower, upper = np.full(initial.size, -np.inf), np.full(initial.size, np.inf)
        for j, i in enumerate(algebraic):
            key, name = names[i].rsplit('.', 1)
            if name == 'P':
                lower[j] = 1.
                scale[j] = max(abs(initial[j]), 1e5)
                if isinstance(self.nodes.get(key), CombustorComponent):
                    ambient = self.nodes[key].definition['ambient_node']
                    lower[j] = self.nodes[ambient].evaluate({}, boundary_values=self._boundary(ambient, self.time))['P']
            elif name.startswith('T'):
                lower[j], scale[j] = 1., 300.
            elif name in ('gas_area_fraction', 'quality'):
                lower[j], upper[j], scale[j] = 0., 1., 1.
            elif name.startswith('mdot'):
                scale[j] = max(abs(initial[j]), .01)
            node = self.nodes.get(key)
            if isinstance(node, (VolumeComponent, JunctionComponent)):
                if node.mode == 'saturated' and name == 'P':
                    lower[j], upper[j] = self.fluid_properties.saturation_bounds(node.fluid_name)
                elif node.mode != 'saturated':
                    fluids = ((node.definition['liquid_fluid'], 'T_liq'), (node.definition['gas_fluid'], 'T_ull')) if node.mode == 'two_phase' else ((node.fluid_name, 'T'),)
                    for fluid, temperature in fluids:
                        pb, tb = self.fluid_properties.state_bounds(fluid)
                        if name == 'P':
                            lower[j], upper[j] = max(lower[j], pb[0]), min(upper[j], pb[1])
                        elif name == temperature:
                            lower[j], upper[j] = max(lower[j], tb[0]), min(upper[j], tb[1])
        for j, i in enumerate(differential, start=len(algebraic)):
            scale[j] = max(abs(self.y[i]), .01)
        x0 = np.maximum(np.minimum(initial, upper), lower) / scale

        def equations(x):
            y, dy = self.y.copy(), self.ydot.copy()
            y[algebraic] = x[:len(algebraic)] * scale[:len(algebraic)]
            dy[differential] = x[len(algebraic):] * scale[len(algebraic):]
            out = np.empty_like(y)
            self.residual(self.time, y, dy, out)
            return out

        result = least_squares(equations, x0, bounds=(lower / scale, upper / scale),
                               xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
        if np.max(np.abs(result.fun)) > self.options['residual_tolerance']:
            worst = list(self.equation_index)[int(np.argmax(np.abs(result.fun)))]
            raise RuntimeError(f'Bounded initialization failed at {worst}: {result.message}')
        self.y[algebraic] = result.x[:len(algebraic)] * scale[:len(algebraic)]
        self.ydot[differential] = result.x[len(algebraic):] * scale[len(algebraic):]

    def _new_session(self, *, seed=False):
        if seed or not self.differential.any():
            self.close()
            self._seed_consistent()
        margins = self._event_margins(self.time, self.y, self.ydot)
        self.root_names = tuple(sorted(margins))
        # Reuse allocations only when the ID vectors, roots and Jacobian
        # structure still describe these equations. Boundary values may change;
        # IDAReInit resets history and CalcIC still restores consistency.
        signature = (tuple(self.variable_index), tuple(self.equation_index), tuple(self.differential),
                     tuple(self.atol), tuple(self.row_scales), self.root_names,
                     tuple((key, getattr(node, 'mode', None)) for key, node in self.nodes.items()),
                     tuple((key, branch.active, self.directions[key], getattr(branch, 'phase', None),
                            tuple(getattr(branch, 'phases', ()))) for key, branch in self.branches.items()),
                     tuple(self.regulator_modes.items()), bool(self.axial_specific_force),
                     self.options['jacobian'], self.options['verify_jacobian'], self.options['suppress_algebraic_error'])
        if self.session is not None and getattr(self, '_session_signature', None) == signature:
            step = self.session.restart(IdaStep(self.time, self.y, self.ydot, np.empty(0, dtype=int), 0),
                                        horizon=self.options['initialization_horizon'],
                                        rtol=self.effective_rtol, max_step=self.effective_max_step)
            self.y, self.ydot = step.y, step.ydot
            return
        self.close()
        self.session = IdaSession(self.residual, self.y, self.ydot, self.differential,
                                  time=self.time, rtol=self.effective_rtol, atol=self.atol,
                                  roots=self.root_values if self.root_names else None,
                                  root_directions=(-1,) * len(self.root_names),
                                  suppress_algebraic_error=self.options['suppress_algebraic_error'] and self.differential.any())
        if self.options['jacobian'] == 'colored':
            install_jacobian(self.session, self, verify=self.options['verify_jacobian'])
        if np.isfinite(self.effective_max_step):
            self.session._check(self.session.idas.IDASetMaxStep(self.session.mem, self.effective_max_step), 'max step')
        step = self.session.consistent(self.options['initialization_horizon'])
        self.y, self.ydot = step.y, step.ydot
        self._session_signature = signature

    def _guesses_for_modes(self, values):
        result = {}
        for key, component in {**self.nodes, **self.branches}.items():
            old = values[key]
            result[key] = {}
            for name in component.variable_names:
                if name in old:
                    value = old[name]
                elif name.startswith('mdot_'):
                    value = max(old.get('mdot', 0.01), 1e-8) / max(len(getattr(component, 'phases', ())), 1)
                elif name == 'gas_area_fraction':
                    value = 0.5
                elif name == 'P':
                    value = component.definition['P0']
                else:
                    raise ValueError(f'Missing event mapping for {key}.{name}')
                result[key][name] = value
        return result

    def _prepare_modes(self, values):
        for _ in range(10):
            self.y = self._build_layout(self._guesses_for_modes(values))
            values = self._unpack(self.y)
            result = self.evaluate_trial(self.time, self.y, select_modes=True)
            values = self._unpack(self.y)
            if result is not None:
                return
        raise RuntimeError('Dependent component modes did not settle')

    def initialize(self, *, time=0.0, bcs=None, heat_rate=None):
        if self.y is not None:
            raise ValueError('Network is already initialized')
        checkpoint = self._checkpoint()
        failures = []
        for attempt in range(self.options['residual_retries'] + 1):
            try:
                result = self._initialize_once(time=time, bcs=bcs, heat_rate=heat_rate)
                self.retry_diagnostics.extend(failures)
                return result
            except ResidualAcceptanceError as error:
                if attempt == self.options['residual_retries']:
                    self.restore(checkpoint)
                    error.attempts = failures + [dict(diagnostic=str(error), exhausted=True)]
                    raise
                floor = min(self.options['retry_rtol_floor'], self.effective_rtol)
                self.effective_rtol = max(floor, self.effective_rtol * self.options['retry_rtol_factor'])
                failures.append(dict(diagnostic=str(error), initialization_retry=True,
                                     rtol=self.effective_rtol, max_step=self.effective_max_step))
            except SolverConvergenceError as error:
                horizon = self.options['initialization_horizon']
                self.restore(checkpoint)
                if not error.retryable or attempt == self.options['residual_retries']:
                    error.attempts = failures + [dict(diagnostic=str(error), exhausted=True)]
                    raise
                self.options['initialization_horizon'] = horizon * .5
                failures.append(dict(diagnostic=str(error), initialization_retry=True,
                                     horizon=self.options['initialization_horizon']))
            except Exception:
                self.restore(checkpoint)
                raise

    def _initialize_once(self, *, time=0.0, bcs=None, heat_rate=None):
        if self.y is not None:
            raise ValueError('Network is already initialized')
        if not np.isfinite(time):
            raise ValueError('Initial time must be finite')
        checkpoint = self._checkpoint()
        self.time, self.bcs, self.heat_rate = float(time), deepcopy(bcs or {}), deepcopy(heat_rate or {})
        values = {key: node.initial_values() for key, node in self.nodes.items()}
        for key, branch in self.branches.items():
            values[key] = {'mdot': float(branch.parameters.get('mdot0', branch.parameters.get('design_mdot', 0.01)))}
            if key in self.regulator_modes:
                p = values[branch.to_node]['P']
                self.regulator_modes[key] = 'regulating' if abs(p - branch.target) <= branch.target * 1e-10 else 'closed' if p > branch.target else 'capacity'
        try:
            self._prepare_modes(values)
            self.ydot = np.zeros_like(self.y)
            self._new_session()
            self._accept()
            if self.stop_at_shutdown and any(isinstance(n, CombustorComponent) and n.mode == 'shutdown' for n in self.nodes.values()):
                self._freeze('engine_shutdown')
            self._settle_initial_events()
            self._accept()
        except Exception:
            self.close()
            self.__dict__ = checkpoint
            self.session = None
            raise
        return self._output()

    def _settle_initial_events(self):
        for _ in range(30):
            if self.frozen:
                return
            margins = self._event_margins(self.time, self.y, self.ydot)
            events = [key for key, value in margins.items() if value < -self._settling_tolerance(key)
                      or (value <= 0 and key[2] in ('dryout', 'switch', 'oxidizer_unavailable', 'fuel_unavailable'))]
            if not events:
                return
            self.apply_events(self.time, self.y, self.ydot, events)
        raise RuntimeError('Initial/event mode iteration did not settle')

    def _settling_tolerance(self, event):
        """Pressure roundoff must not immediately undo a capacity/closure event."""
        tolerance = self.options['event_tolerance']
        if event[0] == 'branch' and event[2] == 'regulate':
            branch = self.branches[event[1]]
            index = self.variable_index[f'{branch.to_node}.P']
            tolerance = max(tolerance, self.options['regulator_pressure_tolerance'],
                            self.options['rtol'] + self.atol[index] / branch.target)
        return tolerance

    def apply_events(self, t, y, ydot, events):
        """Atomically map modes/inventories, initialize and validate a restart."""
        checkpoint = self._checkpoint()
        try:
            self._apply_events(t, y, ydot, events)
            self._accept()
        except Exception:
            self.close()
            self.__dict__ = checkpoint
            self.session = None
            raise

    def _apply_events(self, t, y, ydot, events):
        self.time = float(t)
        before, event_start = deepcopy(self.state), len(self.events)
        terminal = [event for event in events if event[2].startswith('triple_point:')]
        if terminal:
            for kind, key, name in sorted(set(terminal)):
                self.events.append(dict(time_s=t, kind=kind, component=key, name=name))
            self._freeze('triple_point')
            self._record_shutdown(before, event_start)
            return
        values = self._unpack(y)
        for kind, key, name in sorted(set(events)):
            record = {'time_s': self.time, 'kind': kind, 'component': key, 'name': name}
            if kind == 'node':
                node, old = self.nodes[key], values[key]
                if name == 'dryout':
                    if node.mode != 'two_phase':
                        continue
                    remainder = dict(time_s=t, node=key, fluid=node.definition['liquid_fluid'],
                                     mass=old['m_liq'], energy=old['U_liq'])
                    if remainder['mass'] < 0 or remainder['mass'] > 1.01 * node.dry_mass:
                        raise RuntimeError('Dryout mapping exceeded the numerical remainder budget')
                    self.remainders.append(remainder)
                    record['numerical_remainder'] = deepcopy(remainder)
                    values[key] = dict(m=old['m_ull'], U=old['U_ull'], P=old['P'], T=old['T_ull'])
                    node.set_mode('gas')
                elif name in ('oxidizer_unavailable', 'fuel_unavailable'):
                    node.set_mode('shutdown', (), reason=name)
                elif name in ('condense', 'evaporate', 'liquid_limit'):
                    if name == 'condense':
                        values[key] = dict(m=old['m'], U=old['U'], P=old['P'], quality=1.)
                        node.set_mode('saturated')
                    else:
                        sat = node.fluid_properties.saturation_at_p(node.fluid_name, old['P'])
                        values[key] = dict(m=old['m'], U=old['U'], P=old['P'], T=sat.T)
                        node.set_mode('gas' if name == 'evaporate' else 'liquid')
                else:
                    raise RuntimeError(f'Property/model domain limit at t={t:g}: {key}.{name}')
            else:
                branch = self.branches[key]
                if name == 'switch':
                    record['was_open'] = branch.is_open
                    branch.set_open(not branch.is_open)
                    record['is_open'] = branch.is_open
                elif name == 'reverse':
                    self.directions[key] *= -1
                elif name in ('regulate', 'close_limit', 'capacity_limit'):
                    self.regulator_modes[key] = {'regulate': 'regulating', 'close_limit': 'closed', 'capacity_limit': 'capacity'}[name]
                else:
                    raise ValueError(f'Unknown event: {kind}.{key}.{name}')
            self.events.append(record)
        self._prepare_modes(values)
        self._record_shutdown(before, event_start)
        self.ydot = np.zeros_like(self.y)
        if self.stop_at_shutdown and any(isinstance(n, CombustorComponent) and n.mode == 'shutdown' for n in self.nodes.values()):
            # Evaluate only storage reports; a stopped system needs no post-event
            # hydraulic solve, and must not consume another interval's inventory.
            mapped = self._unpack(self.y)
            for key, node in self.nodes.items():
                if isinstance(node, VolumeComponent):
                    self.state.nodes[key] = node.output_state(node.evaluate(mapped[key], axial_specific_force=self.axial_specific_force))
            self._freeze('engine_shutdown')
            return
        self._new_session(seed=any(isinstance(n, CombustorComponent) and n.mode == 'shutdown' for n in self.nodes.values()))
        after_nodes, _, _ = self.evaluate_trial(self.time, self.y)
        # IDACalcIC must not change a conserved inventory during any restart.
        for key, node in self.nodes.items():
            for name in node.differential_variable_names:
                if name in values[key] and after_nodes[key][name] != values[key][name]:
                    raise RuntimeError(f'Restart changed inventory {key}.{name}')

    def _record_shutdown(self, before, event_start):
        """Keep the left-hand endpoint before shutdown erases thrust/flow."""
        for key, node in self.nodes.items():
            if (isinstance(node, CombustorComponent) and node.mode == 'shutdown'
                    and before.nodes[key]['mode'] == 'combusting'):
                record = next((e for e in self.events[event_start:]
                               if e['component'] == key and e['name'] == 'shutdown'), None)
                if record is None:
                    record = dict(time_s=self.time, kind='node', component=key,
                                  name='shutdown', reason=node.shutdown_reason)
                    self.events.append(record)
                record['before'] = dict(
                    node={k: n.as_dict() for k, n in before.nodes.items()},
                    branch={k: b.as_dict() for k, b in before.branches.items()},
                    mdot={k: b.mdot for k, b in before.branches.items()})

    def _accept(self):
        if self.frozen:
            return
        nodes, branches, adjacent = self.evaluate_trial(self.time, self.y)
        # IDA returns interpolated derivatives at roots. At a square-root flow
        # crossing those can have a larger defect than y itself. Reconstruct
        # instantaneous inventory rates from the unchanged conservation laws;
        # do not alter inventories or IDA's internal history.
        for key, node in self.nodes.items():
            names = node.differential_variable_names
            if not names:
                continue
            heat = self._heat_for(key, nodes[key], self.time)
            zero = dict.fromkeys(names, 0.)
            rhs = -node.residual(nodes[key], zero, adjacent[key], heat_rate=heat)[:len(names)]
            matrix = np.column_stack([node.residual(nodes[key], dict(zip(names, column)))[:len(names)]
                                      for column in np.eye(len(names))])
            rates = np.linalg.solve(matrix, rhs)
            for name, rate in zip(names, rates):
                self.ydot[self.variable_index[f'{key}.{name}']] = rate
        residual = np.empty_like(self.y)
        self.residual(self.time, self.y, self.ydot, residual)
        worst = int(np.argmax(np.abs(residual)))
        if not np.isfinite(residual).all() or abs(residual[worst]) > self.options['residual_tolerance']:
            name = list(self.equation_index)[worst]
            raise ResidualAcceptanceError(f'Accepted residual {name}={residual[worst]:.3e} exceeds tolerance')
        for key, branch in self.branches.items():
            if key in self.regulator_modes and self.regulator_modes[key] == 'regulating':
                error = abs(nodes[branch.to_node]['P'] / branch.target - 1.)
                if error > max(20 * self.options['rtol'], 1e-6):
                    raise RuntimeError(f'Regulator pressure drift at {key}: {error:.3e}')
            if 'gas_area_fraction' in branches[key].trial_values:
                fraction = branches[key]['gas_area_fraction']
                if not -1e-8 <= fraction <= 1 + 1e-8:
                    raise RuntimeError(f'Invalid nozzle area split at {key}')
        self.state = NetworkState({k: n.output_state(nodes[k]) for k, n in self.nodes.items()}, deepcopy(branches))
        for key, state in self.state.nodes.items():
            state.properties['heat_rate'] = self._heat_for(key, nodes[key], self.time)
        powered = any(isinstance(node, CombustorComponent) and node.mode == 'combusting'
                      and nodes[key].get('mdot_fuel', 0.) > 0
                      and nodes[key].get('mdot_oxidizer', 0.) > 0
                      for key, node in self.nodes.items())
        self.pressure_tracking.accept(self.time, nodes, powered)
        if self.constraint_monitor:
            for name, margin in self.constraint_monitor(deepcopy(self.state)).items():
                if margin < self.constraints.get(name, np.inf):
                    self.constraints[name], self.constraint_times[name] = margin, self.time

    def _freeze(self, reason):
        self.close()
        self.frozen, self.stop_reason = True, reason
        for key, node in self.nodes.items():
            if isinstance(node, CombustorComponent):
                node.set_mode('shutdown', (), reason=node.shutdown_reason or reason)
                record = self.state.nodes[key]
                record.properties.update(mode='shutdown', shutdown_reason=node.shutdown_reason,
                                         cstar=0., Cf=0., mdot_oxidizer=0., mdot_fuel=0.)
        for record in self.state.branches.values():
            record.enabled = False
            record.flows = {}
            for name in record.trial_values:
                if name.startswith('mdot'):
                    record.trial_values[name] = 0.
            record.properties.update(thrust=0., pump_active=False)

    def _checkpoint(self):
        # Providers can hold native resources; share these read-only services.
        memo = {id(p): p for p in (self.fluid_properties, self.combustion_properties, self.constraint_monitor) if p is not None}
        return deepcopy({k: v for k, v in self.__dict__.items() if k != 'session'}, memo)

    def update(self, dt=None, bcs=None, heat_rate=None, commit=True, axial_specific_force=0.0):
        duration = 0.0 if dt is None else float(dt)
        if not np.isfinite(duration) or duration < 0 or not np.isfinite(axial_specific_force):
            raise ValueError('Require finite nonnegative duration and finite acceleration')
        if not commit:
            preview = copy(self)
            preview.__dict__ = self._checkpoint()
            preview.session = None
            try:
                return preview.update(duration, bcs, heat_rate, True, axial_specific_force)
            finally:
                preview.close()
        checkpoint = self._checkpoint()
        rtol, max_step = self.effective_rtol, self.effective_max_step
        failures = []
        for attempt in range(self.options['residual_retries'] + 1):
            try:
                output = self._advance_interval(duration, bcs, heat_rate, axial_specific_force)
                self.retry_diagnostics.extend(failures)
                return output
            except Exception as error:
                diagnostic = f'Fluid network failed at t={self.time:g}: {error}'
                retry = ((isinstance(error, ResidualAcceptanceError) or
                          isinstance(error, SolverConvergenceError) and error.retryable) and duration > 0
                         and attempt < self.options['residual_retries'])
                self.restore(checkpoint)
                self.last_diagnostic = diagnostic
                if not retry:
                    error.attempts = failures + [dict(diagnostic=diagnostic, exhausted=True)]
                    raise
                if isinstance(error, ResidualAcceptanceError):
                    floor = min(self.options['retry_rtol_floor'], rtol)
                    rtol = max(floor, rtol * self.options['retry_rtol_factor'])
                max_step = min(max_step, duration * .1 * .5**attempt)
                self.effective_rtol, self.effective_max_step = rtol, max_step
                failures.append(dict(diagnostic=diagnostic, rtol=rtol, max_step=max_step))

    def restore(self, checkpoint):
        """Restore a reusable checkpoint; discarded native history is never copied."""
        self.close()
        memo = {id(p): p for p in (self.fluid_properties, self.combustion_properties,
                                   self.constraint_monitor) if p is not None}
        self.__dict__ = deepcopy(checkpoint, memo)
        self.session = None

    def _advance_interval(self, duration, bcs, heat_rate, axial_specific_force):
        if self.frozen:
            self.time += duration
            return self._output()
        changed = ((bcs is not None and bcs != self.bcs)
                   or (heat_rate is not None and heat_rate != self.heat_rate)
                   or axial_specific_force != self.axial_specific_force)
        self.axial_specific_force = float(axial_specific_force)
        if bcs is not None:
            self.bcs = deepcopy(bcs)
        if heat_rate is not None:
            self.heat_rate = deepcopy(heat_rate)
        if self.y is None:
            self.initialize(time=self.time, bcs=self.bcs, heat_rate=self.heat_rate)
        elif self.session is None or changed:
            self._new_session()
            self._settle_initial_events()
        endpoint, start_events = self.time + duration, len(self.events)
        while self.time < endpoint and not self.frozen:
            step = self.session.advance(endpoint)
            self.time, self.y, self.ydot = step.time, step.y, step.ydot
            if step.roots.any():
                self._accept()  # Record the limiting pre-event state as well.
                events = [self.root_names[i] for i in np.flatnonzero(step.roots)]
                for key, node in self.nodes.items():
                    if isinstance(node, PropellantTankComponent) and node.mode == 'two_phase':
                        i = self.variable_index[f'{key}.m_liq']
                        margin = self.y[i] - node.dry_mass
                        tolerance = min(.01 * node.dry_mass,
                                        max(10 * self.atol[i], abs(self.ydot[i]) * self.options['event_time_tolerance']))
                        if self.ydot[i] < 0 and margin <= tolerance:
                            events.append(('node', key, 'dryout'))
                # SUNDIALS reports coincident roots; mode preparation handles
                # downstream phase/chamber changes caused by those events.
                self.apply_events(self.time, self.y, self.ydot, events)
                self._settle_initial_events()
                if len(self.events) - start_events > self.options['max_events']:
                    raise RuntimeError('Event budget exceeded; possible mode cycling')
            self._accept()
        if self.frozen:
            self.time = endpoint
        self._accept()
        return self._output()

    def _output(self):
        state = deepcopy(self.state)
        return dict(time=self.time, state=state,
                    frozen=self.frozen, stop_reason=self.stop_reason,
                    node={k: v.as_dict() for k, v in state.nodes.items()},
                    branch={k: v.as_dict() for k, v in state.branches.items()},
                    td_state={k: v.as_dict() for k, v in state.nodes.items() if isinstance(self.nodes[k], VolumeComponent)},
                    mdot={k: v.mdot for k, v in state.branches.items()},
                    events=deepcopy(tuple(self.events)), numerical_remainders=deepcopy(tuple(self.remainders)),
                    event_counts=dict(Counter(f"{e['kind']}:{e['component']}:{e['name']}" for e in self.events)),
                    constraints=dict(self.constraints), constraint_times=dict(self.constraint_times))

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
