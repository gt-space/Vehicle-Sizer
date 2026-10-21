"""Benchmark the production solver; pre-refactor baseline/harness live at 5b84921."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import wraps
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Configs.loader import load_config
from Flight.Flight import FlightSim
from Fluids.FluidNetwork import FluidNetwork
from Fluids.Sundials.ida_session import IdaSession
import simulation
from simulation_types import history_samples


def instrument():
    records = defaultdict(lambda: dict(calls=0, wall_s=0.0))
    def wrap(cls, name, label):
        original = getattr(cls, name)
        @wraps(original)
        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                value = original(*args, **kwargs)
                return value
            finally:
                rec = records[label]
                rec['calls'] += 1
                rec['wall_s'] += time.perf_counter() - started
        setattr(cls, name, measured)
    for cls, methods in (
        (FluidNetwork, {'update': 'fluid_update', 'restore': 'fluid_restore',
                        '_new_session': 'fluid_restart', '_checkpoint': 'fluid_checkpoint'}),
        (IdaSession, {'advance': 'idas_advance', 'consistent': 'idas_consistency'}),
        (FlightSim, {'run': 'flight', 'rhs': 'flight_rhs',
                     'trial_aero': 'aero', 'step_coupled': 'fluid_thermal'}),
    ):
        for method, label in methods.items():
            wrap(cls, method, label)
    return records


def summarize(result):
    fields = ('termination', 'apogee', 'apogee_time', 'burn_duration', 'final_mass',
              'initial_mass', 'max_q', 'max_aoa_deg', 'final_x', 'final_vx',
              'final_pitch_angle', 'rail_exit_time', 'shutdown_reason')
    return {name: getattr(result, name) for name in fields}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['production'], required=True)
    p.add_argument('--config', default='Configs/flight_pressure_fed_regulator.yaml')
    p.add_argument('--coupling-dt', type=float, default=0.5)
    p.add_argument('--rtol', type=float, default=1e-7)
    p.add_argument('--flight-max-step', type=float, default=2.0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = load_config(ROOT / args.config)
    cfg['environment']['wind']['enabled'] = True
    records = instrument()
    cfg['simulation']['dt'] = args.coupling_dt
    cfg.setdefault('advanced', {}).setdefault('flight', {}).update(
        rtol=args.rtol, max_step=args.flight_max_step)
    last = [0.0]
    def progress(kin):
        now = time.perf_counter()
        if now - last[0] > 10:
            print(f'{args.mode}: t={kin.t:.3f}s h={kin.h:.1f}m elapsed={now-started:.1f}s', flush=True)
            last[0] = now
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(mode=args.mode, config=args.config, wind=True,
                  coupling_dt=args.coupling_dt,
                  baseline_dt=cfg['simulation']['dt'], rk_rtol=args.rtol,
                  rk_max_step=args.flight_max_step, compute_loads=False,
                  record_history=True, python=sys.version,
                  config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())
    source_paths = [ROOT / 'simulation.py', ROOT / 'simulation_types.py']
    for directory in ('Flight', 'Fluids', 'Thermals', 'Vehicle', 'Configs', 'AeroTables', 'FluidTables'):
        source_paths.extend(sorted((ROOT / directory).rglob('*.py')))
    report['source_hashes'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in source_paths}
    started, cpu = time.perf_counter(), time.process_time()
    try:
        result = simulation.simulate(cfg, record_history=True, compute_loads=False, progress=progress)
        report.update(success=True, metrics=summarize(result),
                      accepted_samples=len(result.history), fluid_retries=len(result.fluid_retries),
                      harness=getattr(result, 'harness_diagnostics', {}))
        rows = []
        samples = list(history_samples(result))
        for state in samples:
            kin, fluid = state['kinematics'], state['plant'].fluids
            rows.append([kin.t, kin.h, kin.x, kin.vx, kin.vz, kin.theta, kin.q,
                         state['mass_properties']['total_mass'], state['mass_properties']['Iyy'], fluid.propulsion.thrust, state['atmosphere'].q])
        report['metrics']['impulse_Ns'] = float(np.trapezoid(
            [row[9] for row in rows], [row[0] for row in rows]))
        shutdown = next((s for s in result.history if 'before_event' in s
                         and s['before_event']['engine_on'] and not s['engine_on']), None)
        if shutdown is not None:
            before = shutdown['before_event']['kinematics']
            report['metrics']['burnout_pitch_deg'] = float(np.degrees(before.theta))
            report['metrics']['burnout_vz'] = float(before.vz)
            report['metrics']['burnout_altitude'] = float(before.h)
        np.savetxt(args.output.with_suffix('.csv'), rows, delimiter=',',
                   header='t,h,x,vx,vz,theta,pitch_rate,mass,Iyy,thrust,dynamic_pressure', comments='')
    except Exception as error:
        report.update(success=False, error=str(error), traceback=traceback.format_exc())
        diagnostic = getattr(error, 'diagnostic', None)
        if diagnostic is not None:
            report['diagnostic'] = diagnostic
        traceback.print_exc()
    finally:
        report.update(wall_s=time.perf_counter()-started, cpu_s=time.process_time()-cpu,
                      timings=dict(records))
        args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ('traceback', 'diagnostic', 'source_hashes')}, indent=2), flush=True)
    if not report['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
