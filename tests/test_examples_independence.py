"""Core profiling must work without the examples package."""
from pathlib import Path
import subprocess
import sys


def test_core_profiling_works_with_examples_imports_disabled():
    # Run separately because profiling wraps model methods for the process lifetime.
    script = '''
import sys
sys.modules['examples'] = None
sys.modules['examples.search_timing'] = None
from optimizer.workers import fingerprints
from optimizer.timing import install
from Configs.loader import load_config
from simulation import simulate
sources = fingerprints()
assert 'optimizer/timing.py' in sources
assert 'optimizer/core.py' in sources
assert 'optimizer/workers.py' in sources
assert 'optimizer/__main__.py' in sources
assert 'diagnostics/errors.py' in sources
assert 'diagnostics/constraints.py' in sources
assert 'diagnostics/warnings.py' in sources
assert 'reporting/run_report.py' in sources
assert not any(name.startswith('examples/') for name in sources)
cfg = load_config('Configs/Vespula.yaml')
cfg['simulation']['t_end'] = .1
timing = install()
result = simulate(cfg, record_history=False, compute_loads=False)
assert result.final_time == .1
assert timing.records['flight.total']['calls'] == 1
assert timing.records['setup.vehicle']['wall_s'] > 0
'''
    result = subprocess.run([sys.executable, '-c', script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
