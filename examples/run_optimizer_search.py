"""Example: run a bounded optimizer search through the public Python API."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Configs.loader import load_config
from optimizer.core import optimize
from simulation import project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='Configs/optimizer_epump.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-evaluations', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=300.)
    parser.add_argument('--candidates-per-worker', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    settings = load_config(project_path(args.config))
    settings['max_evaluations'] = args.max_evaluations
    summary = optimize(settings, args.output, workers=args.workers, timeout=args.timeout,
                       candidates_per_worker=args.candidates_per_worker,
                       resume=args.resume, profile=args.profile)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
