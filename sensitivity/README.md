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
`fin_can.span`, or `vehicle.sections[1].masses.cube`. Values use the base
configuration's units (normally SI). Specify `values: [...]`,
`start`/`stop`/`step`, or `start`/`stop`/`num`. One-at-a-time sweeps vary each
parameter separately; `mode: full_grid` evaluates every combination. Both
include a baseline. `max_cases` limits the number of candidate evaluations.

`workers: 1` runs serially; a positive integer enables that many processes,
and `auto` uses up to four. Each worker loads its own property/aero tables.
The usual dependencies and lookup files used by `main.py` are required.

`ignore_wind: true` disables wind only in candidate copies, including the
baseline. `ignore_feasibility: true` bypasses selected design limits and the
configured angle-of-attack limit while retaining their margins in results.
Geometry checks, aerodynamic table bounds, and solver errors still apply.
Leave it false to preserve normal checks. Completion and feasibility are
reported separately.

Set `record_history: true` for time-history partials and end-of-burn fluid
states; `history_detail: full` includes all exposed scalar state fields.
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
results.plot_history('thrust_N', group_by=parameter, save=True, show=False)
results.plot_case_history(y='thrust_N', y2='altitude_m',
                          y2scale=1000, save=True, show=False)
results.plot_derivative(parameter, 'apogee', wrt=parameter, save=True, show=False)
results.plot_history_derivative('altitude_m', wrt=parameter, save=True, show=False)
```

Scalar plots accept a field name, a list of fields, or a mapping from field names
to legend labels for `y` and `y2`. Numeric `xscale`, `yscale`, and `y2scale`
divide displayed values; derivatives retain their underlying data units.
Plots save under the results directory's `plots/` folder when `save=True`.
Use `save=Path(...)` for a specific destination and `show=True` for a window.

For full-grid results, fix the other sweep parameters with
`filters={'other.parameter': value}`. Error cases are excluded; cases marked
`infeasible` are excluded unless `include_infeasible=True`. Sweep runs that
ignored feasibility can mark a case completed while `feasible` remains false;
inspect saved feasibility fields when that distinction matters.

The command-line entry point is `python -m reporting.plot_sensitivities` (or
`python reporting/plot_sensitivities.py`). It supports `--list-fields`,
`--mode scalar|history|derivative|history-derivative`, `--x`, `--y`, `--y2`,
`--parameter`, repeated `--filter PARAMETER=VALUE`, axis labels and display
divisors, `--output`, and `--show`. Run it with `--help` for usage.
