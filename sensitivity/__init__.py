"""Optional vehicle sensitivity sweeps; normal Vehicle-Sizer entry points unchanged."""
from .parameters import PARAMETER_PATHS, apply_case, make_cases, sweep_values
from .results import SweepResults
from .runner import run_file, run_sweep

__all__ = ['PARAMETER_PATHS', 'SweepResults', 'apply_case', 'make_cases',
           'run_file', 'run_sweep', 'sweep_values']
