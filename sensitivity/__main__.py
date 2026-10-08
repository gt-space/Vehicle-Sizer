"""python -m sensitivity Configs/sweeps/vespula_sweep.yaml [--dry-run]"""
from __future__ import annotations
import argparse
from .runner import run_file


def main():
    parser = argparse.ArgumentParser(description='Run or validate Vehicle-Sizer sensitivity sweeps')
    parser.add_argument('config', help='Sweep YAML with base_config and sweep.parameters')
    parser.add_argument('--dry-run', action='store_true', help='Validate and enumerate cases without flight simulation')
    args = parser.parse_args()
    result = run_file(args.config, dry_run=args.dry_run)
    if not args.dry_run:
        print(f'Sweep saved to {result.folder}')


if __name__ == '__main__':
    main()
