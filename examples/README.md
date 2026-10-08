# Examples

These are standalone commands that demonstrate the public APIs. Main application
modules do not import or fingerprint anything in this folder. Run from the project
root.

Propulsion at fixed ambient pressure, with JSON output:

```bash
.venv/bin/python examples/run_propulsion.py Configs/Vespula.yaml --duration 1 --dt 0.1
```

The input must use table properties and tank heating set to `None` or `DensityPowerLaw`.

A 100-evaluation optimizer search through `optimizer.core.optimize()`:

```bash
.venv/bin/python examples/run_optimizer_search.py --config Configs/optimizer_epump.yaml --output outputs/example_search_01 --workers 4 --max-evaluations 100
```

Add `--resume` to repeat the command with the original output directory, settings,
and evaluation budget. For normal optimizer use, run `python -m optimizer`.
Worker supervision and profiling are implemented by `optimizer/workers.py` and
`optimizer/timing.py`.

Profile a short simulation and save nested wall/CPU timing records:

```bash
.venv/bin/python examples/search_timing.py Configs/Vespula.yaml --duration 1 --output outputs/example_timings.json
```

The timing example calls `simulation.simulate()` with the optional core profiler.
It can be edited or removed without affecting main optimizer operation.
