"""Example: profile a standalone flight simulation and save nested CPU/wall timings."""
import argparse
import json
from math import isfinite
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Configs.loader import load_config
from optimizer.timing import install
from simulation import project_path, simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default='Configs/Vespula.yaml')
    parser.add_argument('--duration', type=float, default=1.)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/example_timings.json')
    args = parser.parse_args()
    if not isfinite(args.duration) or args.duration <= 0:
        parser.error('duration must be finite and positive')
    cfg = load_config(project_path(args.config))
    cfg['simulation']['t_end'] = args.duration
    timing = install()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = simulate(cfg, record_history=False, compute_loads=False)
    finally:
        timing.save(args.output)
    print(json.dumps(dict(termination=result.termination, final_time_s=result.final_time,
                          timings_file=str(args.output), timing_categories=len(timing.records)), indent=2))


if __name__ == '__main__':
    main()
