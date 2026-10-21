"""Flight-only interpolation, adaptive integration and dense event location."""
from bisect import bisect_left, bisect_right
from dataclasses import replace

import numpy as np
from scipy.integrate import RK45
from scipy.optimize import brentq

from diagnostics.errors import TrialDomainError
from simulation_types import KinematicsState


def sample_interval(t, samples, side='right'):
    """Interpolate propulsion and cached mass moments, never thermodynamic states.

    Samples are (time, FluidOut, mass_properties) tuples. Duplicate times delimit
    discontinuities; left selects the incoming mode, right the outgoing mode.
    """
    times = [sample[0] for sample in samples]
    index = (bisect_left(times, t) if side == 'left' else bisect_right(times, t)) - 1
    index = max(0, min(index, len(samples) - 1))
    ta, a, ma = samples[index]
    tb, b, mb = samples[min(index + 1, len(samples) - 1)]
    w = np.clip((t - ta) / (tb - ta), 0., 1.) if tb > ta else 0.
    mix = lambda x, y: x + w * (y - x)
    mass = mix(ma['total_mass'], mb['total_mass'])
    cg = mix(ma['total_mass'] * ma['cg'], mb['total_mass'] * mb['cg']) / mass
    inertia = mix(ma['Iyy'] + ma['total_mass'] * ma['cg']**2,
                  mb['Iyy'] + mb['total_mass'] * mb['cg']**2) - mass * cg**2
    properties = dict(ma, total_mass=mass, cg=cg, Iyy=inertia)
    if 'Ixx' in ma:
        properties['Ixx'] = mix(ma['Ixx'], mb['Ixx'])
    propulsion = replace(a.propulsion, **{
        key: mix(getattr(a.propulsion, key), getattr(b.propulsion, key))
        for key in ('thrust', 'Pc', 'MR', 'Cf', 'cstar', 'mdot_ox', 'mdot_fuel', 'mdot_nozzle')})
    return propulsion, properties


def advance_flight(kin, stops, rhs, rail, rail_height, settings):
    """Return accepted flight states and events without committing any model.

    Fluid event times are hard step boundaries. Flight roots use the accepted
    RK45 step's dense polynomial; a mode switch restarts only the flight solver.
    """
    accepted = []
    atol = [settings['position_atol']] * 2 + [settings['velocity_atol']] * 2
    atol += [settings['angle_atol'], settings['angular_rate_atol']]
    tolerance = settings['event_time_tolerance']
    for stop in stops:
        def derivative(t, y):
            state = KinematicsState(t, *y)
            return rhs(state, rail, 'left' if t == stop else 'right')

        def solver_at(state, first_step=None):
            return RK45(derivative, state.t,
                        [state.x, state.h, state.vx, state.vz, state.theta, state.q],
                        stop, rtol=settings['rtol'], atol=atol,
                        max_step=settings['max_step'], first_step=first_step)

        # Supplying the first step avoids SciPy's unbounded initial-state probe.
        solver = solver_at(kin, min(.01, settings['max_step'], stop - kin.t))
        retries = 0
        while solver.status == 'running':
            try:
                message = solver.step()
            except TrialDomainError as error:
                step = min(solver.h_abs, settings['max_step'], stop - kin.t) / 2
                if retries >= settings['trial_max_retries'] or step < settings['trial_min_dt']:
                    raise RuntimeError(f'Flight trial recovery exhausted at t={kin.t:g}: {error}') from error
                retries += 1
                solver = solver_at(kin, step)
                continue
            if solver.status == 'failed':
                raise RuntimeError(message)
            retries = 0
            dense = solver.dense_output()
            endpoint = dense(solver.t)
            events = []
            for name, index, threshold, crossed in (
                ('rail_release', 1, rail_height, rail and kin.h < rail_height <= endpoint[1]),
                ('apogee', 3, 0., kin.vz > 0. and endpoint[3] <= 0.),
            ):
                if crossed:
                    time = brentq(lambda t: dense(t)[index] - threshold, kin.t, solver.t,
                                  xtol=tolerance, maxiter=settings['event_max_iterations'])
                    events.append((time, name))
            # A shortened plant solve may end a few ulps before the same root.
            # Accept its ascent-side limit within the requested time tolerance.
            if kin.vz > 0 and 0 <= endpoint[3] <= max(0., -solver.f[3]) * tolerance:
                events.append((solver.t, 'apogee'))
            time, event = min(events) if events else (solver.t, None)
            kin = KinematicsState(time, *dense(time))
            if event == 'apogee':
                kin = replace(kin, vz=0.)
            accepted.append((kin, rail, event))
            if event == 'apogee':
                return accepted
            if event == 'rail_release':
                rail = False
                if kin.t < stop:
                    solver = solver_at(kin, min(.01, settings['max_step'], stop - kin.t))
    return accepted
