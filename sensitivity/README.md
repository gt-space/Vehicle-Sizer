# Vehicle-Sizer sensitivity sweeps

This optional module runs independent variations of an existing vehicle YAML via the unchanged `simulation.simulate()` function. The regular `main.py` and optimizer continue to work as before.

## 1. Configure

Copy `Configs/sweeps/vespula_sweep.yaml`. Set `base_config` to your normal flight YAML and choose `mode: one_at_a_time` or `mode: full_grid`. **Every supported parameter is listed**, and only those marked `enabled: true` are varied. Disabled parameters retain their **original** values (their sample range/list is ignored). Each enabled parameter accepts exactly one of:

```yaml
chamber_pressure:
  enabled: true
  values: [1800000, 2000000, 2200000]
```

```yaml
chamber_pressure:
  enabled: true
  start: 1800000
  stop: 2200000
  step: 100000
```

```yaml
chamber_pressure:
  enabled: true
  start: 1800000
  stop: 2200000
  num: 5
```

`stop` is included when it lies exactly on the step. With `num`, endpoints are always included. All values are in the units already used by the vehicle YAML (SI, with thrust tilt in degrees). **No unit tags or conversions.** Invalid definitions and unsupported template pressure references fail during validation rather than silently doing nothing.

`throat_area` and `target_thrust` cannot both be enabled. `throat_area` sets the physical throat size and disables `thrust_target` within that candidate, but preserves prescribed `design_mdot_*` values. `target_thrust` sets `prop_system.thrust_target`, removes the candidate's fixed throat area **and** explicit `design_mdot_oxidizer` and `design_mdot_fuel`, and lets the existing `PropSystem` engine sizing calculate throat and design mass flows. Hardware CdAs, nozzle expansion ratio, aerodynamic exit-diameter override, engine mass, vehicle inertia override, and other explicit design assumptions stay fixed; these may limit physical consistency or cause infeasible cases. The original YAML is never modified. Compare `design_summary.design_thrust` and achieved `thrust_N` against the requested target.

For additional numeric fields, use their existing dotted config paths in `parameters` (e.g. `vehicle.OMLD`), provided that they already exist and are scalar numbers. The direct path is not a guarantee of a compatible physically meaningful model response.

## 2. Run

Run from repository root in the existing project virtual environment (which needs numpy, matplotlib, scipy, PyYAML, h5py, and the usual Vehicle-Sizer dependencies):

```bash
python -m sensitivity Configs/sweeps/vespula_sweep.yaml --dry-run
python -m sensitivity Configs/sweeps/vespula_sweep.yaml
```

`--dry-run` validates the entire sweep and prints the generated cases **without** loading fluid or drag lookup tables. A full run requires `FluidTables/sizer_lookups.h5`, `AeroTables/dragmodel.h5`, and any wind CSV referenced by the chosen vehicle YAML. The two HDF5 lookup files are normally supplied separately from source control.

- `one_at_a_time`: baseline plus one changed input at a time.
- `full_grid`: baseline plus Cartesian product of all enabled input value lists. `max_cases` guards against accidentally huge runs.
- Each run starts from a fresh deep-copy and uses the existing solver. A rejected design is recorded as `infeasible`, other exceptions as `error` with tracebacks. Successful flight results have status `completed`, which does **not** imply that apogee or all mission constraints were achieved.
- Reusable property and aerodynamic models are created once for a normal serial sweep; fresh mutable propulsion/flight state is created for each case.

Generated directory:

```text
outputs/sensitivity/vespula/
    sweep_config.yaml
    baseline_config.yaml
    cases.csv
    summary.csv
    errors.csv
    histories.h5
```

The summary records independent sweep inputs, all numeric `SimResult` scalar fields, and nested numeric design/constraint diagnostics. `histories.h5` stores the numeric synchronized flight history columns from the existing `main.history_rows()` function. It retains individual time arrays and only stores histories for completed/rejected evaluations when recorded data exist. Text-only history columns are not plotted. `record_history: false` reduces memory and output but disables history plotting and design-summary extraction. Flight histories can be large for fine timesteps and many cases.

## 3. Plotting API (implemented entirely inside `sensitivity/`)

`plot_sensitivities.py` at the repository root is **only a test/example**. You can import
`SweepResults` anywhere in your own code, choose which output variables to graph, and
set the plot title, axis labels, line styling, limits, saving and display independently.
No modification to `plot_sensitivities.py` is required when using the API elsewhere.

```python
from sensitivity import SweepResults

results = SweepResults.load('outputs/sensitivity/vespula')
print(results.fields())            # All saved scalar fields
print(results.history_fields())    # All saved flight-history fields

# Scalar output versus an input sweep parameter
fig = results.plot(
    x='chamber_pressure', y='apogee',
    title='Chamber pressure vs. apogee',
    xlabel='Chamber pressure (Pa)', ylabel='Apogee (m)',
    kind='line', marker='o', linewidth=2,
    save=True,    # automatic file in outputs/sensitivity/vespula/plots/
)

# Flight-history comparison
results.plot_history(
    y='thrust_N', group_by='chamber_pressure',
    title='Engine thrust during flight',
    xlabel='Time (s)', ylabel='Thrust (N)',
    save='outputs/sensitivity/vespula/plots/thrust.png',
)

# Derivative of any scalar output with respect to an independent sweep input
results.plot_derivative(
    x='chamber_pressure', y='apogee', wrt='chamber_pressure',
    ylabel='d(Apogee)/d(Pc) (m/Pa)', save=True,
)

# Partial derivative of a time-dependent result
results.plot_history_derivative(
    y='altitude_m', wrt='chamber_pressure', at=2000000,
    title='Altitude sensitivity vs. flight time', save=True,
)

# Output vs. output: displays interactively, without saving
results.plot(
    x='burn_duration', y='apogee', kind='scatter',
    title='Burn duration vs. apogee',
    xlabel='Burn duration (s)', ylabel='Apogee (m)',
    save=False, show=True,
)
```

### Common plotting options

All four methods accept:

| Argument | Behavior |
| --- | --- |
| `title` | Figure title |
| `xlabel`, `ylabel` | Human-readable axis labels (defaults are raw result field names) |
| `save=False` (or omitted) | No file written; plot is shown when created on fresh axes |
| `save=True` | Automatically save a PNG in `<results folder>/plots/` |
| `save='path/to/name.png'` | Save to a specific path (relative paths use the process working directory) |
| `show=True/False` | Explicitly open/hide Matplotlib's interactive window (default: show only unsaved plots created by the call) |
| `figsize=(8, 5)`, `dpi=160` | Figure size in inches and saved image resolution |
| `xlim`, `ylim` | Plot bounds, e.g. `xlim=(0, 300)` |
| `xscale`, `yscale` | Axis scale: e.g. `'linear'` or `'log'` |
| `grid`, `legend`, `legend_title` | Grid and legend formatting |
| `ax=existing_ax` | Draw onto an existing Matplotlib axis for custom/combined figures |
| `filters={'cf_efficiency': 0.90}` | Fix other swept parameters for one-dimensional full-grid comparisons |
| `include_infeasible=True` | Also plot results from infeasible candidates, if they have valid data |

`results.plot(...)` and `results.plot_derivative(...)` additionally accept `kind='line'` or
`kind='scatter'`; the scalar plot defaults to a line for independent sweep inputs and
scatter for output-vs-output comparisons. Standard Matplotlib style keywords such as
`color`, `marker`, `linewidth`, `linestyle`, `alpha`, and `label` pass through to the
underlying line or scatter call. `results.plot_history(...)` accepts `case_labels` to
customize legend entries by case ID or sweep value. All methods **return a Matplotlib
`Figure`**, so you can keep customizing it with Matplotlib when needed.

For multiple figures, pass `show=False` and call `matplotlib.pyplot.show()` after you
have created them. The package does not force an image-only Matplotlib backend.
For scripts run without a graphical environment, set `show=False`.

Numeric derivatives can also be accessed without plotting:
`results.scalar_derivative('apogee', 'chamber_pressure')`,
`results.history_derivative('altitude_m', 'chamber_pressure')`, and
`results.history('case_00001')`. An output-versus-output plot shows correlation, not
a controlled partial derivative.

For a full grid, provide `filters` fixing **every other enabled parameter** when
plotting an individual input or derivative:

```python
results.plot(x='chamber_pressure', y='apogee',
             filters={'cf_efficiency': 0.90},
             title='Apogee at Cf efficiency = 0.90')
```

Numerical partial derivatives are computed from **available independent parameter
values**, keeping the other sweep parameters fixed. Central differences are used
inside the sampled domain and one-sided differences at boundaries. With two points,
these are secants. Time-history derivatives interpolate over a common observed time
window and never extrapolate past flight termination. At discontinuities (e.g. engine
shutdown) numerical derivatives may be undefined. Infeasible cases are excluded
by default.

## 4. Caveats

- **COPV pressure** changes tank design pressure, also used for initial pressurant state by the supplied templates. It does not remove explicit mass overrides or update other hardware limitations.
- **Tank material** and **burn duration** are not sweep controls. `burn_duration` is an available scalar output.
- **Thrust tilt** is `engine.thrust_tilt_deg`, not engine translational offset.
- **Weld allowable** is `advanced.weld_allowable` [Pa], not efficiency.
- **Thrust-target redesign** is limited by fixed existing hardware assumptions. Report both design and achieved thrust. The module does not rewrite the physical propulsion or aerodynamic equations.
- No special treatment is applied to categorical fields or arbitrary arbitrary-component geometry. Dotted config paths must refer to existing numeric values.
- This serial first implementation does not currently support parallel workers, automatic resume, or cached derivative HDF5 output; derivatives are recalculated cheaply from persisted sweep data.
