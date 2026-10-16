"""Run fixed-ambient propulsion with prescribed heating, but no aeroheating wall solver."""
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
from Fluids.design import prepare_design_inputs
from Vehicle.Vehicle import Vehicle
from FluidTables.PropertyModels import property_sources
from Thermals.heat_sources import thermal_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', type=Path,
                        default=ROOT / 'Configs/flight_pump_fed_regulator_epump.yaml')
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
    pure, combustion = property_sources(cfg)
    template, states, wall_pressures = prepare_design_inputs(cfg)
    vehicle = Vehicle(cfg, pure, initial_states=states, wall_pressures=wall_pressures)
    with PropSystem(cfg, vehicle.tanks, pure, combustion,
                    template=template, initial_states=states) as propulsion:
        if any(thermal_model(node.get('thermal')) == 'Aeroheating'
               for node in propulsion.node_definitions.values()):
            parser.error('Aeroheating requires the flight/wall solver; use None or DensityPowerLaw here')
        atm = SimpleNamespace(p=args.ambient_pressure)
        propulsion.update(None, atm, {})
        while propulsion.network.time < args.duration:
            out = propulsion.update(min(args.dt, args.duration - propulsion.network.time), atm, {})
            print(json.dumps(dict(time_s=propulsion.network.time, **asdict(out.propulsion),
                                  events=[{k: v for k, v in event.items() if k != "before"}
                                          for event in out.events], constraints={key: record.margin for key, record in out.constraint_records.items()})), flush=True)


if __name__ == '__main__':
    main()
