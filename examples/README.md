# Examples

These are standalone commands that demonstrate the public APIs. Main application
modules do not import or fingerprint anything in this folder. Run from the project
root.

Propulsion at fixed ambient pressure, with JSON output:

```bash
.venv/bin/python examples/run_propulsion.py Configs/Vespula.yaml --duration 1 --dt 0.1
```

The input must use table properties and tank heating set to `None` or `Prescribed`.

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

Profile canard costs or compare fluid/RK timestep settings (pressure-fed, wind on):

```bash
.venv/bin/python examples/benchmark_canards.py --profile --output outputs/canard_profile.json
.venv/bin/python examples/benchmark_canards.py --dt 1 --max-step 5 --output outputs/canard_steps.json
```

The default horizon is 100 s, before this configuration's known late-coast aero
domain limit. Runs omit history and structural loads, but save accepted flight
states and roll traces beside the JSON. Use `--mode base` or `--mode roll` to
isolate feature costs. `--rtol` and `--atol-scale` vary RK error tolerances;
`--max-step` is only a cap. Profile times include nested calls and must not be
summed; compare runtimes from runs without `--profile`.

Compare the production 240-point command search with an experimental Brent solve:

```bash
.venv/bin/python examples/benchmark_canards.py --command-solver grid --output outputs/command_grid.json
.venv/bin/python examples/benchmark_canards.py --command-solver brent --output outputs/command_brent.json
```

The bracketed variant uses a 1e-10 rad command tolerance and checks saturation
before solving. It replaces the method only inside the harness process; production
code is unchanged. Run cases sequentially in separate processes, repeat with
alternating order, and compare metrics as well as timings. The direct solve removes
the old command-curve interpolation error, so residual roll need not be identical.
