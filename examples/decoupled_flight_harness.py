"""Experimental multirate flight driver used only by benchmark_decoupling.py.

Scope: the supplied pressure-fed, no-external-wall-heating configuration with
fluid_solve_post_shutdown=False. Fluid boundaries are held at predicted endpoint
pressure and starting axial force, matching the production prediction policy.
Coupling accuracy is assessed by interval refinement, not an outer iteration.
Only flight-required outputs are interpolated; this is not a general FluidOut
interpolator or a replacement for production reporting/structural loads.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np
from scipy.integrate import RK45
from scipy.optimize import brentq

from Flight.Flight import FlightSim
from Flight.flight_forces import gravity
from diagnostics.constraints import merge_records
from diagnostics.errors import TrialDomainError
from simulation_types import FluidOut, KinematicsState, SimResult


@dataclass
class Knot:
    t: float
    fluid: FluidOut
    mass: float
    first_moment: float
    inertia_origin: float
    Ixx: float
    axial_mass: np.ndarray


class Interval:
    def __init__(self, knots):
        # Stable sort preserves left/right states at repeated event times.
        self.knots = sorted(knots, key=lambda k: k.t)
        self.times = np.array([k.t for k in self.knots])

    def sample(self, t, side='right'):
        if t < self.times[0] - 1e-8 or t > self.times[-1] + 1e-8:
            raise ValueError(f'Sample {t} outside [{self.times[0]}, {self.times[-1]}]')
        exact = np.flatnonzero(np.abs(self.times - t) < 1e-10)
        if exact.size:
            a = b = self.knots[exact[0] if side == 'left' else exact[-1]]
            w = 0.0
        else:
            j = np.searchsorted(self.times, t, side='right')
            a, b = self.knots[j-1], self.knots[j]
            w = (t-a.t)/(b.t-a.t)
        mix = lambda x, y: x + w * (y-x)
        mass = mix(a.mass, b.mass)
        cg = mix(a.first_moment, b.first_moment) / mass
        inertia = mix(a.inertia_origin, b.inertia_origin) - mass*cg*cg
        propulsion = replace(a.fluid.propulsion, **{
            name: mix(getattr(a.fluid.propulsion, name), getattr(b.fluid.propulsion, name))
            for name in ('thrust', 'Pc', 'MR', 'Cf', 'cstar', 'mdot_ox', 'mdot_fuel', 'mdot_nozzle')})
        # Node thermodynamic reports are deliberately not interpolated. The
        # benchmark exports kinematics/thrust/mass only; no structural loads.
        fluid = replace(a.fluid, propulsion=propulsion, events=())
        return fluid, mass, cg, inertia, mix(a.Ixx, b.Ixx), a, b, w


class DecoupledFlight(FlightSim):
    coupling_dt = 2.0
    rk_rtol = 1e-7
    rk_max_step = 0.25

    def _knot(self, t, fluid):
        self.vehicle.update_mass_distribution(fluid.node)
        mass, cg = float(self.vehicle.total_mass), float(self.vehicle.cg)
        return Knot(t, fluid, mass, mass*cg, float(self.vehicle.Iyy)+mass*cg*cg,
                    float(self.vehicle.Ixx), self.vehicle.mass.copy())

    def _network_fluid(self):
        output = self.prop_system.network._output()
        return FluidOut(node=output['node'], branch=output['branch'], mdot=output['mdot'],
                        propulsion=self.prop_system._propulsion_output(output),
                        event_counts=output['event_counts'],
                        constraint_records=output['constraint_records'])

    def _fluid_interval(self, kin, fluid, forces, duration):
        start = self._knot(kin.t, fluid)
        predicted = self.predict_kinematics(replace(kin, dt=duration), forces)
        # Only altitude/airspeed are needed for this configuration's pressure
        # boundary. Do not require an extrapolated pitch to pass an aero check.
        atm = self._atmosphere(predicted)
        network = self.prop_system.network
        knots = [start]
        accept = network._accept
        def collect():
            accept()
            knots.append(self._knot(float(network.time), self._network_fluid()))
        network._accept = collect
        try:
            end = self.update_fluid(duration, atm, commit=True,
                                    axial_specific_force=forces['axial_specific_force'])
        finally:
            del network._accept
        knots.append(self._knot(kin.t+duration, end))
        self.diag['coupling_intervals'] += 1
        self.diag['fluid_events'].extend({key: event.get(key) for key in
            ('time_s', 'kind', 'component', 'event')} for event in end.events)
        return Interval(knots), end, predicted, atm

    def _state(self, t, y, interval, rail, side='right', dt=1.0, record=False):
        fluid, mass, cg, inertia, Ixx, a, b, w = interval.sample(t, side)
        self.vehicle.cg = cg
        self.vehicle.total_mass = mass
        self.vehicle.Iyy = inertia
        self.vehicle.Ixx = Ixx
        if record:
            self.vehicle.mass = a.axial_mass + w*(b.axial_mass-a.axial_mass)
        kin = KinematicsState(t=t, dt=max(dt, 1e-12), x=y[0], h=y[1], vx=y[2], vz=y[3],
                              theta=y[4], q=y[5], alpha=0., m=mass, Iyy=inertia)
        kin = self._kinematic_boundary(kin, rail)
        return kin, fluid

    @staticmethod
    def _vector(kin):
        return np.array([kin.x, kin.h, kin.vx, kin.vz, kin.theta, kin.q])

    def _record(self, kin, fluid, rail, before=None):
        result = self.result
        evaluation = self.evaluate_state(kin, fluid)
        atmosphere, plant, forces = evaluation
        self._record_extrema(result, atmosphere, plant.aero, kin)
        self._record_twr(result, kin, forces['twr'])
        result.max_altitude = max(result.max_altitude, kin.h)
        result.final_time, result.final_altitude = kin.t, kin.h
        result.final_velocity = result.final_vz = kin.vz
        result.final_x, result.final_vx = kin.x, kin.vx
        result.final_speed = math.hypot(kin.vx, kin.vz)
        result.final_pitch_angle, result.final_pitch_rate = kin.theta, kin.q
        result.final_mass = kin.m
        state = self._collect_sample(result, kin, *evaluation, rail, False)
        if before is not None and state is not None:
            state['before_event'] = before
        if state is not None:
            result.history.append(state)
        return evaluation

    def _integrate(self, kin, interval, rail, progress):
        end_time = interval.times[-1]
        # Every fluid sample is an integration boundary, including duplicate
        # left/right event knots; interpolation never crosses a mode change.
        stops = sorted(set(t for t in interval.times if t > kin.t+1e-10))
        rail_end = self.cfg['launch']['altitude'] + self.cfg['launch']['rail_height']
        for stop in stops:
            while kin.t < stop-1e-10:
                def rhs(t, y):
                    self.diag['rk_rhs_calls'] += 1
                    side = 'left' if abs(t-stop) < 1e-10 else 'right'
                    stage, fluid = self._state(t, y, interval, rail, side)
                    _, _, force = self.evaluate_state(stage, fluid)
                    if rail:
                        return np.array([0., stage.vz, 0., force['az'], 0., 0.])
                    return np.array([stage.vx, stage.vz, force['ax'], force['az'],
                                     stage.q, force['pitch_acceleration']])
                try:
                    solver = RK45(rhs, kin.t, self._vector(kin), stop, rtol=self.rk_rtol,
                                  atol=[1e-5, 1e-5, 1e-6, 1e-6, 1e-9, 1e-9],
                                  max_step=self.rk_max_step)
                except TrialDomainError:
                    # SciPy's automatic initial-step selection also probes a
                    # future state. Bypass that probe and let step recovery
                    # handle any further invalid stages.
                    self.diag['flight_domain_retries'] += 1
                    h = .5*min(self.rk_max_step, stop-kin.t)
                    solver = RK45(rhs, kin.t, self._vector(kin), stop, rtol=self.rk_rtol,
                                  atol=[1e-5, 1e-5, 1e-6, 1e-6, 1e-9, 1e-9],
                                  first_step=h, max_step=h)
                restart = False
                while solver.status == 'running':
                    previous = kin
                    old_t, old_y = solver.t, solver.y.copy()
                    nfev = solver.nfev
                    try:
                        message = solver.step()
                    except TrialDomainError:
                        # Rebuild from the last accepted flight state. No fluid
                        # rollback. Restrict the trial until within aero domain.
                        self.diag['flight_domain_retries'] += 1
                        h = min(solver.h_abs, self.rk_max_step, stop-old_t) * .5
                        if h < 1e-7:
                            raise
                        solver = RK45(rhs, old_t, old_y, stop, rtol=self.rk_rtol,
                                      atol=[1e-5, 1e-5, 1e-6, 1e-6, 1e-9, 1e-9],
                                      first_step=h, max_step=h)
                        continue
                    if solver.status == 'failed':
                        raise RuntimeError(message)
                    self.diag['rk_accepted_steps'] += 1
                    self.diag['rk_rejected_steps'] += max(0, (solver.nfev-nfev)//6-1)
                    t, y = solver.t, solver.y.copy()
                    dense = solver.dense_output()
                    event = None
                    if rail and old_y[1] < rail_end <= y[1]:
                        t = brentq(lambda s: dense(s)[1]-rail_end, old_t, t, xtol=1e-10)
                        y, event = dense(t), 'rail_release'
                    elif old_y[3] > 0 and y[3] <= 0:
                        t = brentq(lambda s: dense(s)[3], old_t, t, xtol=1e-10)
                        y, event = dense(t), 'apogee'
                        y[3] = 0.0
                    side = 'left' if abs(t-stop) < 1e-10 else 'right'
                    kin, fluid = self._state(t, y, interval, rail, side,
                                             t-previous.t, record=True)
                    if self.engine_on(interval.sample(previous.t)[0].propulsion):
                        self.result.burn_duration += t-previous.t
                    before = None
                    if event == 'rail_release':
                        before = self._collect_sample(self.result, kin,
                                  *self.evaluate_state(kin, fluid), rail, False)
                        rail = False
                        self.result.rail_exit_time = t
                        kin = self._kinematic_boundary(kin, False)
                        restart = True
                    self._record(kin, fluid, rail, before)
                    if progress:
                        progress(kin)
                    if event == 'apogee':
                        self.result.apogee = kin.h
                        self.result.apogee_time = kin.t
                        self.result.termination = 'apogee'
                        return kin, rail
                    if restart:
                        break
                if not restart:
                    break
            # Right side of any fluid event at this boundary.
            left = interval.sample(stop, 'left')[0]
            right = interval.sample(stop, 'right')[0]
            kin, _ = self._state(stop, self._vector(kin), interval, rail, 'right', record=True)
            if left.propulsion.mode != right.propulsion.mode:
                before = self.result.history.pop() if self.result.history else None
                self._record(kin, right, rail, before)
        return kin, rail

    def run(self, h0=0.0, v0=0.0, *, progress=None, record_history=True, compute_loads=False):
        started = time.perf_counter()
        if self.thermal is not None or self.cfg['simulation'].get('fluid_solve_post_shutdown', True):
            raise ValueError('Harness requires no external wall heating and fluid_solve_post_shutdown=False')
        if compute_loads:
            raise ValueError('Harness does not interpolate structural load inputs')
        if not record_history:
            raise ValueError('Benchmark requires history for accuracy comparison')
        if min(self.coupling_dt, self.rk_rtol, self.rk_max_step) <= 0:
            raise ValueError('Benchmark step sizes and tolerances must be positive')
        self.diag = dict(coupling_intervals=0, rk_rhs_calls=0, rk_accepted_steps=0,
                         rk_rejected_steps=0, flight_domain_retries=0,
                         fluid_events=[],
                         max_pressure_prediction_relative_error=0.,
                         max_axial_force_prediction_error=0.)
        self.fluid_solve_post_shutdown = False
        rail = self.on_rail(h0)
        kin = KinematicsState(t=0., dt=self.coupling_dt, x=0., h=h0, vx=0., vz=v0,
                              theta=math.pi/2, q=0., alpha=0.,
                              m=float(self.vehicle.total_mass), Iyy=float(self.vehicle.Iyy))
        kin = self._kinematic_boundary(kin, rail)
        fluid = self.update_fluid(None, self._atmosphere(kin), commit=True,
                                 axial_specific_force=gravity(kin.m, h0)/kin.m)
        start = self._knot(0., fluid)
        kin = replace(kin, m=start.mass, Iyy=start.inertia_origin-start.first_moment**2/start.mass)
        self.result = SimResult(max_altitude=h0, initial_mass=kin.m, final_mass=kin.m,
                                dry_mass=float(np.sum(self.vehicle.dry_mass)), history=[])
        self.result.harness_diagnostics = self.diag
        initial = self._record(kin, fluid, rail)
        self.result.initial_state = self.result.history.pop()
        if progress:
            progress(kin)
        forces = initial[2]
        t_end = float(self.cfg['simulation']['t_end'])
        while kin.t < t_end and not self.result.apogee_reached:
            if fluid.propulsion.mode == 'shutdown':
                fluid = self.stop_fluids(fluid)
                start = self._knot(kin.t, fluid)
                interval = Interval([start, replace(start, t=t_end)])
                kin, rail = self._integrate(kin, interval, rail, progress)
                break
            duration = min(self.coupling_dt, t_end-kin.t)
            interval, endpoint, predicted, predicted_atm = self._fluid_interval(kin, fluid, forces, duration)
            start_force = forces['axial_specific_force']
            kin, rail = self._integrate(kin, interval, rail, progress)
            if self.result.apogee_reached and endpoint.propulsion.mode != 'shutdown':
                raise RuntimeError('Powered apogee requires terminal fluid rollback; outside harness scope')
            fluid = endpoint
            self.vehicle.update_mass_distribution(fluid.node)
            # Preserve sampled flight mass in the coast apogee case; all
            # inventories are frozen by the network at shutdown.
            atm, _, forces = self.evaluate_state(kin, fluid)
            self.diag['max_pressure_prediction_relative_error'] = max(
                self.diag['max_pressure_prediction_relative_error'],
                abs(atm.p-predicted_atm.p)/max(1., atm.p))
            self.diag['max_axial_force_prediction_error'] = max(
                self.diag['max_axial_force_prediction_error'],
                abs(forces['axial_specific_force']-start_force))
            self.result.burn_complete = fluid.propulsion.mode == 'shutdown'
            self.result.shutdown_reason = fluid.propulsion.shutdown_reason
            merge_records(self.result.constraint_records, fluid.constraint_records)
        self.diag['flight_wall_s'] = time.perf_counter()-started
        return self.result.history
