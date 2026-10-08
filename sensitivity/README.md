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
- `sweep.workers`: positive integer specifying parallel flight-simulation processes. **Default is 1** (serial, same behavior as before). Use `workers: 2` or `workers: 4` to parallelize, or `workers: auto` to use up to four processes, leaving one logical CPU available when possible. Large fluid/engine/drag lookup tables are loaded independently once per worker, so RAM consumption increases with worker count. The same setting works when starting `run_sensitivities.py` with VS Code's Run Python File button, as well as from the command line. `--dry-run` only enumerates cases and does not launch worker processes.
- `sweep.ignore_feasibility`: `false` by default. When `true`, bypass *engineering feasibility rejections* in the sizing/flight pipeline: vehicle maximum length/diameter ratio, pump electric power sizing margin, initial pressurant/propellant tank pressure and temperature minima, and the configured max-AoA flight rejection. The simulation still **calculates and reports** these signed margins. A completed flight is labeled `completed` despite negative margins, but `feasible` and `accepted` retain their actual engineering meanings. Other postflight limits (apogee, burn duration, dynamic pressure, rail T/W, stability) are recorded without stopping a valid flight. **Hard physical/model guards remain**: negative geometric packaging clearances (e.g., an engine/nozzle that will not fit), invalid inputs, out-of-range aerodynamic lookup data, pump-curve flow bounds, thermodynamic property domains, numerical integration failures, and no-apogee outcomes can still stop or fail a candidate. Different pressure-fed/pump-fed/regulator/bang-bang/gas-generator templates use the same controls, but this does **not** guarantee every case in every template can physically finish.
- `sweep.ignore_wind`: `false` by default. When `true`, overrides `environment.wind.enabled: false` on each **temporary simulation candidate**, even if the base vehicle YAML enables wind. The wind file is not loaded for those flights. The original vehicle YAML, saved baseline definition, and ordinary standalone runs are unchanged.

  ```yaml
  sweep:
    mode: one_at_a_time
    workers: 6
    ignore_feasibility: true
    ignore_wind: true
    parameters:
      # Existing parameters, unchanged
  ```

  When `ignore_feasibility` is off, existing feasibility and early-stop behavior is unchanged. In either mode, numerical exceptions remain `error`; incomplete runs are `infeasible`. Existing result files are not retroactively reclassified. These options apply in both serial and parallel workers.
- Each run starts from a fresh deep-copy and uses the existing solver. With the default option, a rejected design or flight with violated constraints is recorded as `infeasible`; exceptions are recorded as `error` with tracebacks.
- With one worker, reusable property and aerodynamic models are created once per sweep. With multiple workers, the **spawn** multiprocessing mode (safe for macOS) constructs lookup models separately within each worker, then reuses those read-only models across its assigned cases. The parent process is the only writer of `histories.h5`/CSV files, preventing concurrent HDF5 writes. Results remain indexed by deterministic case IDs, even if individual cases finish out of order. Each process uses a fresh vehicle, propulsion, and flight state. Numerical results should agree with serial sweeps to normal solver tolerances.
- Start with `workers: 2` and increase to `4` if memory allows. Using all CPU cores can slow the machine when each worker loads large tables or numerical libraries spawn additional native threads. The operating system handles worker scheduling; speed-up depends on per-case runtime, available RAM, and solver/thread overhead. Only one Python process should run a given output directory at a time.

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
- Automatic resume and cached derivative HDF5 output are not implemented; derivatives are recalculated cheaply from persisted sweep data.


## More outputs, multiple traces, and dual y-axes

For each simulation with design reporting, `summary.csv` now exposes the actual
sized-vehicle tank and section measurements as numeric scalar fields:

- `design_summary.tank.fuel_tank.mass` – dry fuel tank mass (including section hardware)
- `design_summary.tank.fuel_tank.shell_mass` – calculated structural shell mass
- `design_summary.tank.ox_tank.mass` / `.shell_mass` – oxidizer tank
- `design_summary.tank.press_tank.mass` / `.shell_mass` – pressurant tank
- `design_summary.tank.total_mass`, `.total_shell_mass`, `.total_hardware_mass`
- `design_summary.tank.<tank_id>.volume`, `.length`, `.diameter`, `.thickness`
- `design_summary.section.<section_id>.mass`, `.length`, `.station` – all sections
- The previously available result fields (`apogee`, `dry_mass`, `max_q`,
  `pump_sizing.*`, `gg_sizing.*`, `load_peaks.*`, and others) remain available.

`mass` is the installed **dry section mass**, including any manual masses attached to
that tank section. `shell_mass` is just its wall/shell mass, calculated by the
sizer. `hardware_mass = mass - shell_mass`. For a welded tank stress sweep,
`shell_mass` is normally the most informative plot. The COPV shell mass need not
vary with weld allowable if its construction uses a fixed empirical mass model.

Inspect the names actually present in your results:

```python
print(results.fields('tank'))       # substring match, case insensitive
print(results.fields('mass'))
print(results.history_fields('pressure'))
```

**To get newly exported tank/section masses, rerun the sweep once** after
updating the package (with `outputs.record_history: true`). Existing results
can already plot multiple lines using the scalar fields they contain.
`outputs.record_history: true` also enables plotting saved time histories.

### Several tank masses on one axis

```python
results.plot(
    x='weld_allowable',
    y={
        'design_summary.tank.fuel_tank.shell_mass': 'Fuel tank shell',
        'design_summary.tank.ox_tank.shell_mass': 'Oxidizer tank shell',
        'design_summary.tank.press_tank.shell_mass': 'COPV shell',
    },
    xscale=6.894757e6,  # Pa to ksi
    xlabel='Weld Allowable (ksi)', ylabel='Shell Mass (kg)',
    title='Tank structural mass versus weld allowable',
    save=True, filename='tank_masses', file_format='png',
)
```

### Apogee (left y-axis) and tank shells (right y-axis)

```python
results.plot(
    x='weld_allowable',
    y={'apogee': 'Apogee'},
    y2={
        'design_summary.tank.fuel_tank.shell_mass': 'Fuel shell',
        'design_summary.tank.ox_tank.shell_mass': 'LOX shell',
    },
    xscale=6.894757e6, yscale=1e3, y2scale=1,
    xlabel='Weld Allowable (ksi)',
    ylabel='Apogee (km)', y2label='Tank Shell Mass (kg)',
    save=True, filename='apogee_and_tank_mass', file_format='png',
)
```

For scalar plots, `y` and `y2` each accept a field string, list of field strings,
or dictionary mapping field name to legend label. A single trace defaults to
black; multiple traces use distinguishable colors and a combined legend.
`labels={'apogee': 'Peak Altitude'}` optionally overrides legend labels.
`ylim`, `yscale` affect the left axis; `y2lim`, `y2scale` affect the right axis.
Numeric scale arguments **divide** displayed values, without altering saved
numerical data. Use `xscale=6.894757e6` for Pa -> ksi, `yscale=1e3` for m -> km.

### Multiple history traces for one flight, including y2

```python
results.plot_case_history(
    case_id='baseline',
    x='time_s',                       # can be another numeric history field
    y={'thrust_N': 'Thrust'},
    y2={'altitude_m': 'Altitude'},
    y2scale=1e3,
    xlabel='Flight Time (s)',
    ylabel='Thrust (N)', y2label='Altitude (km)',
    save=True, filename='baseline_thrust_altitude', file_format='jpeg',
)
```

`plot_case_history` plots fields from a **single** saved flight using two axes.
The existing `plot_history(y, group_by)` is unchanged and still compares one
history quantity across all cases using sweep-value colors and a colorbar.

**Limits:** Non-numeric metadata such as tank material, shutdown reason and
complex solver objects cannot be drawn as continuous numeric curves; they
remain separate from scalar values. The available columns depend on the
propulsion template and enabled output options; requesting a missing field
raises an error that lists available choices. Solver internals not exported
by the simulation aren't automatically accessible.
