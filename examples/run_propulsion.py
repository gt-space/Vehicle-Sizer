"""Run a table-backed SUNDIALS propulsion config at fixed ambient pressure, without flight/thermals."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from Fluids.PropSystem import PropSystem
from Vehicle.Vehicle import Vehicle
from FluidTables.PropertyModels import TablePureFluidPropertySource, TableCombustionPropertySource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', type=Path,
                        default=ROOT / 'Configs/flight_pump_fed_regulator.yaml')
    parser.add_argument('--duration', type=float, default=1.)
    parser.add_argument('--dt', type=float, default=.1)
    parser.add_argument('--ambient-pressure', type=float, default=101325.)
    args = parser.parse_args()
    import math
    if any(not math.isfinite(v) or v <= 0 for v in (args.duration, args.dt, args.ambient_pressure)):
        parser.error('duration, dt and ambient pressure must be finite and positive')
    with args.config.open() as stream:
        cfg = yaml.safe_load(stream)
    if any(cfg['property_models'][name]['source'] != 'table' for name in ('pure_fluid', 'combustion')):
        parser.error('This example expects table-backed pure-fluid and combustion properties')
    pure_cfg = cfg['property_models']['pure_fluid']['table']
    combustion_cfg = cfg['property_models']['combustion']['table']
    pure = TablePureFluidPropertySource(ROOT / pure_cfg['lookup_file'], pure_cfg['fluids'])
    combustion = TableCombustionPropertySource(ROOT / combustion_cfg['lookup_file'], combustion_cfg['nfz'])
    vehicle = Vehicle(cfg, pure)
    with PropSystem(cfg, vehicle.tanks, pure, combustion) as propulsion:
        atm = SimpleNamespace(p=args.ambient_pressure)
        propulsion.update(None, atm, {})
        while propulsion.network.time < args.duration:
            out = propulsion.update(min(args.dt, args.duration - propulsion.network.time), atm, {})
            print(json.dumps(dict(time_s=propulsion.network.time, **asdict(out.propulsion),
                                  events=[{k: v for k, v in event.items() if k != "before"}
                                          for event in out.events], constraints=out.constraints)), flush=True)


if __name__ == '__main__':
    main()
