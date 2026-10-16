# Sensitivity sweeps and plots

Run independent copies of a vehicle configuration with selected input changes.
The runner sizes and simulates each case and saves results for plotting and
numerical partial derivatives. Existing vehicle files are never edited.

## Run a sweep

From the project root:

```bash
.venv/bin/python -m sensitivity Configs/sweeps/copv_volume.yaml --dry-run
.venv/bin/python -m sensitivity Configs/sweeps/copv_volume.yaml
```

The supplied file uses your pressure-fed vehicle, sweeps COPV volume from 60 to
80 L, and includes its original volume as a separate baseline. Edit this sweep
file to choose inputs, values, workers, and the output directory.
For VS Code's Run Python File button, use `run_sensitivities.py`; its
`SWEEP_CONFIG` and `DRY_RUN` settings select the file and validation mode.

Parameter keys are configuration paths, such as `prop_system.Pc_target`,
`fin_can.span`, or `vehicle.sections[1].masses.cube`. A readable label may be used
as the key when its definition supplies an explicit `path`, such as
`copv_volume: {path: tanks.press_tank.volume, values: [0.06, 0.07]}`.
Paths identify tanks directly; parameter names are not translated through aliases.
Values use the base
configuration's units (normally SI). Specify `values: [...]`,
`start`/`stop`/`step`, or `start`/`stop`/`num`. One-at-a-time sweeps vary each
parameter separately; `mode: full_grid` evaluates every combination. Both
include a baseline. `max_cases` limits the number of candidate evaluations.

Use `linked_paths: [...]` to vary several inputs together. Normally all paths
receive the same absolute value. With `offset: true`, values are increments
added to each path's own baseline instead. The plotted parameter is the offset
(baseline zero); `inputs.<path>` columns retain the actual input values.
Offset sweeps require existing numeric fields and validate the resulting values.

## Six-study wind-off suite

```bash
.venv/bin/python examples/run_pressure_fed_sweeps.py \
  --config Configs/flight_pump_fed_regulator_gg.yaml \
  --output outputs/gg_sensitivity_2026-10-09 --workers 4
```

Despite its historical filename, this script supports pressure-fed and pump-fed
configs. It sweeps added mass on each propellant tank (0–20 kg in 2 kg steps),
pressure loss, COPV volume (baseline ±15 L in 3 L steps), diameter (10–15 in in
0.5 in steps), c* efficiency (0.80–1.00 in 0.025 steps), and drag multiplier
(0.8–1.2 in 0.05 steps). Pump-fed pressure sweeps add the same ±15 psi offset
in 5 psi steps to all four tank-to-pump and pump-to-injector losses. Pressure-fed
pressure sweeps use ±30 psi in 5 psi steps on both tank-to-injector losses.
Other inputs remain fixed; the model recalculates sizing for each candidate.
Inputs outside model coverage and solver failures remain recorded as failed cases.

Each run saves individual direct plots, a six-panel direct overview, and a
six-panel derivative figure under its own `plots/` directory (PNG and SVG).
The overview figures have no overall title; direct plots mark the baseline
with a dashed line. Use `--dry-run` to validate or `--plot-only` to regenerate
figures from a saved output directory. Source configs remain unchanged.


`workers: 1` runs serially; a positive integer enables that many processes,
and `auto` uses up to four. Each worker loads its own property/aero tables.
The usual dependencies and lookup files used by `main.py` are required.

`ignore_wind: true` disables wind only in candidate copies, including the
baseline. Sweeps and single runs call `simulate` with the same defaults: no
design-constraint assessment. The optimizer alone opts into assessment.
Sweep definitions validate paths, types and ranges; physical input validity is
checked by the shared simulator for each case, so invalid endpoints do not
prevent the rest of the range from running.
The old `sweep.ignore_feasibility` key has no effect and can be removed.
Model coverage, physical state validity and numerical convergence are always
required. Completed runs may violate optimizer design limits.

A failed case retains its inputs and typed diagnostic in `summary.csv` and
`errors.csv`, but has no scalar flight outputs or saved history. The sweep
continues through every requested case; plots and derivatives skip failed
cases. `warnings.csv` separately records nonfatal warning codes, messages and
counts for successful and failed cases.

Run the same sweep from the main entry point with:

```bash
.venv/bin/python main.py --sweep Configs/sweeps/copv_volume.yaml
```

Set `record_history: true` for time-history partials and end-of-burn fluid
states; `history_detail: full` includes all exposed scalar state fields.
Sweep exports share `reporting.history.capture_history` with the CLI.
History fields use canonical `state.*` or `derived.*` paths; the default time
coordinate is `state.kinematics.t`. Summary statistics use
`history.<field>.<statistic>`; fluid end-of-burn values use the `burnout` statistic
at the same observed shutdown transition as flight burnout. The old short
history names and fluid EOL aliases are no longer exported. Extrema include
pre-event snapshots; fields absent at launch use `first_recorded`, not `initial`.
Structural profiles require both `compute_loads: true` and
`record_structural_profiles: true`. Results include baseline/sweep YAML copies,
case and summary CSVs, an error log, a field catalog, and HDF5 histories.
Reusing an output directory overwrites the previous sweep files; choose a new
directory for runs you want to keep separately.

Python callers can use `from sensitivity import run_file, run_sweep`.
`run_file(path, dry_run=True)` validates inputs and lists cases without flights.


## Plot saved results

A results directory needs `sweep_config.yaml`, `summary.csv`, and `cases.csv`.
History plots additionally need `histories.h5`. `field_catalog.csv` is optional.
If the saved configuration omits `sweep.mode`, results use `one_at_a_time`;
`full_grid` is also supported.

```python
from sensitivity import SweepResults

results = SweepResults.load('outputs/sensitivity/my_sweep')
print(results.fields())
print(results.field_catalog())
print(results.history_fields())

parameter = 'tanks.press_tank.volume'  # Use a parameter from your saved sweep.
results.plot(parameter, 'apogee', y2='dry_mass',
             xscale=0.001, yscale=1000,
             xlabel='COPV volume (L)', ylabel='Apogee (km)',
             y2label='Dry mass (kg)', save=True, show=False)
results.plot_history('state.plant.fluids.propulsion.thrust', group_by=parameter, save=True, show=False)
results.plot_case_history(y='state.plant.fluids.propulsion.thrust', y2='state.kinematics.h',
                          y2scale=1000, save=True, show=False)
results.plot_derivative(parameter, 'apogee', wrt=parameter, save=True, show=False)
results.plot_history_derivative('state.kinematics.h', wrt=parameter, save=True, show=False)
```

Scalar plots accept a field name, a list of fields, or a mapping from field names
to legend labels for `y` and `y2`. Numeric `xscale`, `yscale`, and `y2scale`
divide displayed values; derivatives retain their underlying data units.
Plots save under the results directory's `plots/` folder when `save=True`.
Use `save=Path(...)` for a specific destination and `show=True` for a window.

For full-grid results, fix the other sweep parameters with
`filters={'other.parameter': value}`. Error cases are excluded; cases marked
`infeasible` are excluded unless `include_infeasible=True`. Older saved runs may include feasibility assessments; new sweeps do not assess
design constraints.

The command-line entry point is `python -m reporting.plot_sensitivities` (or
`python reporting/plot_sensitivities.py`). It supports `--list-fields`,
`--mode scalar|history|derivative|history-derivative`, `--x`, `--y`, `--y2`,
`--parameter`, repeated `--filter PARAMETER=VALUE`, axis labels and display
divisors, `--output`, and `--show`. Run it with `--help` for usage.
