# Vehicle Sizer

Builds a rocket from YAML, simulates propulsion and planar flight, calculates tank heating and structural loads, and searches for low launch-mass designs subject to constraints.

Use Python 3.12+ and the project virtual environment. Run commands from the project root.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install numpy scipy PyYAML h5py CoolProp ussa1976 matplotlib sundials4py pytest
```

Inputs are a vehicle configuration file ([config_Name].yaml), its propulsion template (defines how fluid system nodes connect), and the referenced property, aero, and optional wind files. Table configs need `FluidTables/sizer_lookups.h5` and `AeroTables/dragmodel.h5`. Config dimensions use SI. You can find lookup tables here: https://app.notion.com/p/Elytra-Vehicle-Sizing-a6388149302944a9abf85dbf69d75622

To simulate a flight:

```bash
.venv/bin/python main.py Configs/flight_pressure_fed_regulator.yaml
```

Outputs are a terminal summary, flight/event/load CSVs, and PNG plots at the config's `simulation.output` and `simulation.plot` paths. Use `--dt` or `--t-end` to change the run. Single runs and sweeps do not assess design constraints. The optimizer uses the same `simulate` entry point with `assess_constraints=True` and ranks the resulting margins separately from execution failures.

All simulation callers use the same model-domain and solver checks. A run that
cannot reach apogee raises `EvaluationFailure`, with a typed cause, phase and
collected warnings. The single-run CLI prints the reason and writes
`<output>.failure.json`. Warnings are nonfatal; design-limit violations only
affect optimizer assessment. The former `--enforce-constraints` option is removed.

For direct model assembly, call `Fluids.design.prepare_design_inputs(cfg)` after
applying candidate overrides. It returns `(template, states, wall_pressures)`.
Pass `initial_states=states, wall_pressures=wall_pressures` to `Vehicle` and
`template=template, initial_states=states` to `PropSystem`. These constructors
consume prepared inputs; `simulate` handles this preparation automatically.
Construct property sources with `FluidTables.PropertyModels.property_sources(cfg)`
and supply both to `PropSystem`. Source selection uses `property_models` exclusively;
PropSystem no longer creates defaults or reads `engine.property_source`.

Run a sensitivity sweep using copies of your existing vehicle configuration:

```bash
.venv/bin/python main.py --sweep Configs/sweeps/copv_volume.yaml --dry-run
.venv/bin/python main.py --sweep Configs/sweeps/copv_volume.yaml
```

See [sensitivity/README.md](sensitivity/README.md) for sweep inputs, parallel
execution, saved plots, and numerical partial derivatives.

Run an optimization with a new output directory:

```bash
.venv/bin/python -m optimizer Configs/optimizer_pressure_fed.yaml --output outputs/search_01
```

The search writes configs, evaluation records, failure details, and a summary. Accepted designs produce `best.yaml`; successful verification produces `verified.yaml`. Use `--max-evaluations` to limit the search.
Run independent candidates in separate processes with `--workers`:

```bash
.venv/bin/python -m optimizer Configs/optimizer_epump.yaml --output outputs/epump_search_01 --workers 4 --max-evaluations 100
```

Workers reuse property tables and recycle after `--candidates-per-worker` candidates (default 3). `--timeout` limits each candidate's wall time (default 300 seconds). Parallel searches use deferred population updates and can follow a different trajectory from serial searches with the same seed. Resume a worker search by repeating its command with `--resume`; keep the original config, evaluation budget, and source files unchanged. The saved results are replayed before unfinished work continues. The default is one serial worker; `workers`, `worker_timeout`, and `candidates_per_worker` can also be set in the optimizer config.
You need to make an optmiziation configuration specific to your flight config, examples are in Configs folder.

Run tests:

```bash
.venv/bin/python -m pytest -q
```

Simulation finalization attaches design, battery, pump, gas-generator, pressure
tracking and retry data through one path for completed and partial results.
Failure diagnostics retain their execution phase and cause; propulsion resources
close even if metadata finalization fails.

| Folder | Purpose |
| --- | --- |
| [optimizer](optimizer/README.md) | Design searches, process workers, profiling |
| [sensitivity](sensitivity/README.md) | Parameter sweeps, saved histories, plots, partial derivatives |
| [diagnostics](diagnostics/README.md) | Errors, feasibility constraints, warnings |
| [reporting](reporting/README.md) | Flight plots and run summaries |
| [Configs](Configs/README.md) | Flight inputs, search settings, network wiring |
| [Vehicle](Vehicle/README.md) | Geometry, mass distribution, inertia, stiffness |
| [Fluids](Fluids/README.md) | Propulsion sizing and fluid-network solving |
| [Flight](Flight/README.md) | Trajectory, aerodynamic forces, structural loads |
| [Thermals](Thermals/README.md) | Wall temperatures and fluid heat transfer |
| [FluidTables](FluidTables/README.md) | Fluid and combustion property tables |
| [AeroTables](AeroTables/README.md) | Aerodynamic tables and interpolation |
| [examples](examples/README.md) | Standalone propulsion, optimizer, and profiling examples |
| [tests](tests/README.md) | Automated checks |
