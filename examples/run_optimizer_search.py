"""Compatibility entry point for the optimizer worker runner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optimizer_workers import cli

if __name__ == '__main__':
    cli()
