"""Profile canards and compare timestep settings without modifying production physics.

Runs stop at 100 s by default because the current wind-on canard configuration
reaches the aero-domain limit before apogee. Timings exclude loads and history.
Use separate processes for cases; nested profile times must not be added.
"""
import argparse
from collections import defaultdict
from functools import wraps
import hashlib
from importlib import import_module
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Configs.loader import load_config
from diagnostics.errors import EvaluationFailure, SimulationIncompleteError
from Flight.Flight import FlightSim
from Flight.canards import CanardSystem, CanardAerodynamics
from Flight.roll import RollAnalysis
from Fluids.FluidNetwork import FluidNetwork
from simulation import simulate


def bracketed_command_deflection(self, mach, alpha, open_moment, torque_scale, active):
    """Experimental replacement for the production 240-point command curve."""
    limit_key = 'max_measured_angle_deg' if mach <= 2.8 else 'max_extension_angle_deg'
    lift_limit = np.deg2rad(self.config.get(limit_key, 22. if mach <= 2.8 else 15.))
    angle_limit = min(np.deg2rad(self.config.get('max_deflection_deg', 0.)),
                      max(lift_limit - abs(alpha), 0.))

    def coefficient(command):
        return float(self.aero.roll_coefficient(self.geometry, mach, command, alpha))

    maximum_lift = coefficient(angle_limit)
    maximum_torque = torque_scale * maximum_lift
    demand = abs(open_moment)
    if not active or demand == 0:
        return 0., maximum_torque, False
    if maximum_torque <= 0:
        return 0., maximum_torque, True
    required = demand / torque_scale
    if required >= maximum_lift:
        return angle_limit, maximum_torque, required > maximum_lift
    command = brentq(lambda delta: coefficient(delta) - required,
                     0., angle_limit, xtol=1e-10)
    return command, maximum_torque, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['base', 'roll', 'canards'], default='canards')
    parser.add_argument('--dt', type=float, default=.5)
    parser.add_argument('--max-step', type=float, default=2.)
    parser.add_argument('--rtol', type=float, default=1e-7)
    parser.add_argument('--atol-scale', type=float, default=1.)
    parser.add_argument('--duration', type=float, default=100.)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--command-solver', choices=['grid', 'brent'], default='grid',
                        help='Use production command search or experimental bracketed solve')
    parser.add_argument('--skip-stability', action='store_true', help='Diagnostic-cost ablation only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command_solver == 'brent':
        CanardSystem.command_deflection = bracketed_command_deflection
    cfg = load_config(ROOT / 'Configs/flight_pressure_fed_regulator.yaml')
    cfg['simulation'].update(dt=args.dt, t_end=args.duration)
    cfg['environment']['wind']['enabled'] = True
    flight = cfg['advanced']['flight']
    flight.update(max_step=args.max_step, rtol=args.rtol)
    for key, default in dict(position_atol=1e-5, velocity_atol=1e-6,
                             angle_atol=1e-9, angular_rate_atol=1e-9).items():
        flight[key] = default * args.atol_scale
    if args.mode != 'canards':
        cfg.pop('canards')
    if args.mode == 'base':
        cfg.pop('roll')

    timings = defaultdict(lambda: dict(calls=0, inclusive_s=0., exclusive_s=0.))
    stack, trace, counts = [], [], defaultdict(int)
    original_rhs = FlightSim.rhs
    def rhs(self, *values, **kwargs):
        counts['rhs'] += 1
        return original_rhs(self, *values, **kwargs)
    FlightSim.rhs = rhs
    module = import_module('Flight.Flight')
    advance = module.advance_flight
    def tracked_advance(*values, **kwargs):
        counts['flight_intervals'] += 1
        accepted = advance(*values, **kwargs)
        if not trace:
            trace.append(list(vars(values[0]).values()))
        trace.extend(list(vars(kin).values()) for kin, _, _ in accepted)
        return accepted
    module.advance_flight = tracked_advance
    original_update = FluidNetwork.update
    def update(self, *values, **kwargs):
        counts['fluid_updates'] += 1
        return original_update(self, *values, **kwargs)
    FluidNetwork.update = update
    if args.skip_stability:
        original_extrema = FlightSim._record_extrema
        def extrema(self, *values, **kwargs):
            roll, self.roll_analysis = self.roll_analysis, None
            try:
                return original_extrema(self, *values, **kwargs)
            finally:
                self.roll_analysis = roll
        FlightSim._record_extrema = extrema

    def measure(cls, method):
        original = getattr(cls, method)
        label = cls.__name__ + '.' + method
        @wraps(original)
        def measured(*values, **kwargs):
            started = time.perf_counter()
            frame = [0.]
            stack.append(frame)
            try:
                return original(*values, **kwargs)
            finally:
                elapsed = time.perf_counter() - started
                stack.pop()
                if stack:
                    stack[-1][0] += elapsed
                row = timings[label]
                row['calls'] += 1
                row['inclusive_s'] += elapsed
                row['exclusive_s'] += elapsed - frame[0]
        setattr(cls, method, measured)
    if args.profile:
        for cls, methods in (
            (FlightSim, ('run', 'rhs', 'evaluate_state', 'trial_aero', 'step_coupled',
                         '_record_extrema', '_record_roll', 'prepare_samples')),
            (CanardSystem, ('evaluate', 'body_forces', 'command_deflection')),
            (CanardAerodynamics, ('roll_coefficient', 'drag', 'canard_lift', 'shock_expansion_cn')),
            (RollAnalysis, ('forcing', 'sample')),
        ):
            for method in methods:
                measure(cls, method)

    report = dict(settings=vars(args) | {'output': str(args.output)},
                  config=cfg, profile={}, counts={})
    report['source_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in ('Flight', 'Fluids', 'Vehicle') for p in sorted((ROOT/directory).rglob('*.py'))}
    started, cpu = time.perf_counter(), time.process_time()
    result, failure = None, None
    try:
        result = simulate(cfg, record_history=False, compute_loads=False)
    except EvaluationFailure as error:
        result = error.partial_result
        if not isinstance(error.__cause__, SimulationIncompleteError):
            failure = str(error)
    report.update(wall_s=time.perf_counter()-started, cpu_s=time.process_time()-cpu,
                  failure=failure, counts=dict(counts), profile=dict(timings))
    report['reached_horizon'] = result is not None and result.final_time == args.duration
    if result is not None:
        report['metrics'] = {key: getattr(result, key) for key in (
            'final_time', 'final_altitude', 'final_x', 'final_vx', 'final_vz',
            'final_pitch_angle', 'final_pitch_rate', 'final_mass', 'burn_duration',
            'rail_exit_time', 'max_q', 'max_aoa_deg', 'min_stability_calibers')}
        if result.roll:
            roll = result.roll
            report['metrics'].update(final_roll_rate=roll['p'][-1], final_roll_angle=roll['phi'][-1],
                                     max_abs_roll_rate=float(np.max(np.abs(roll['p']))))
        if trace:
            array = np.asarray(trace)
            steps, end_times = np.diff(array[:, 0]), array[1:, 0]
            report['counts']['accepted_steps'] = len(steps)
            report['step_stats'] = {}
            for name, mask in (('powered', end_times <= result.burn_duration),
                               ('coast', end_times > result.burn_duration)):
                values = steps[mask]
                if values.size:
                    report['step_stats'][name] = dict(count=len(values),
                        minimum=float(values.min()), median=float(np.median(values)),
                        maximum=float(values.max()), at_cap=int(np.sum(np.isclose(values, args.max_step))))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output.with_suffix('.npz'), flight=array,
                **({'roll_time': result.roll['time'], 'roll_rate': result.roll['p'],
                    'roll_angle': result.roll['phi']} if result.roll else {}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_hashes','config','profile')}, indent=2), flush=True)
    if not report['reached_horizon'] or failure:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
