# Vehicle-Sizer sensitivity sweeps

The sensitivity runner creates **independent, resized vehicle candidates** from an
existing vehicle YAML, simulates each candidate, and writes portable CSV and HDF5
results. The baseline file is never modified. This implementation is generic: it
does not maintain a long registry of every possible tank, pump, valve, airframe,
or engine input.

## 1. Minimal generic sweep YAML

```yaml
base_config: Configs/flight_pressure_fed_regulator.yaml
sweep:
  mode: one_at_a_time
  workers: 6
  ignore_feasibility: true
  ignore_wind: true
  parameters:
    tanks.press_tank.volume:
      start: 0.05
      stop: 0.12
      step: 0.005
outputs:
  record_history: true
  history_detail: standard
  compute_loads: false
  directory: outputs/sensitivity/flight_pressure_fed_regulator
```

**No `enabled` flag is needed.** A parameter is swept if listed. Parameters
absent from the file retain their baseline values. The optional legacy
`enabled: true/false` setting and old names (`copv_volume`, `chamber_pressure`,
`weld_allowable`, etc.) remain compatible, but new files should use paths.

Supported value declarations:

```yaml
parameters:
  tanks.press_tank.volume:
    values: [0.06, 0.08, 0.10]
  tanks.press_tank.design_pressure:
    start: 20000000
    stop: 30000000
    step: 2000000
  advanced.weld_allowable:
    start: 100000000
    stop: 200000000
    num: 11
```

**All values use the original configuration units**, with no implicit unit
conversion. `stop` is included when exactly divisible by `step`; `num`
includes both endpoints. Values that equal the original input are omitted
from a one-at-a-time sweep because the standalone baseline already captures
them. `full_grid` evaluates every Cartesian combination plus a separate baseline;
`sweep.max_cases` limits potential work. Case order and IDs are deterministic.

### Other generic inputs

Dictionary paths and list indices both work:

```yaml
parameters:
  tanks.ox_tank.propellant_mass:
    values: [260, 280, 300]
  prop_system.initial_conditions.press_tank.T:
    values: [270, 290, 310]
  vehicle.sections[1].masses.counterweight:
    values: [0, 5, 10]
  fin_can.fin_count:
    values: [3, 4, 5]   # integer values stay integers
```

You can also supply explicit `path:` and an abbreviated, user-chosen name:

```yaml
parameters:
  copv_start_temperature:
    path: prop_system.initial_conditions.press_tank.T
    values: [280, 300, 320]
```

Simple categorical and boolean inputs support `values: [...]` (not numeric
`step` or `num`). This can select existing materials or settings, but only
values valid for the underlying solver are physically meaningful. Paths must
exist and reference scalar inputs. Known virtual sizing inputs
`engine.throat_area`, `prop_system.thrust_target`, and `engine.thrust_tilt_deg`
can be created when their alternate modes/defaults permit it.

### Template-local parameters

Propulsion network templates are read from `prop_system.template`. If the base
config selects a template file and a path begins with
`prop_system.template.branches.` or `.nodes.`, the runner loads the template
**into the temporary candidate** and changes its numeric values there. For example,
when a template actually contains a numeric `CdA`:

```yaml
parameters:
  prop_system.template.branches.MY_VALVE.CdA:
    values: [0.00001, 0.00002]
```

Template field names/branch IDs vary by architecture; check the selected
`Configs/templates/*.yaml` before using a path. Do not try to sweep a
`{config: some.path}` template reference object: instead sweep the referenced
vehicle `some.path` itself. No source template file is edited.

### Feasibility and input conflicts

`sweep.ignore_feasibility: true` bypasses engineered sizing and flight constraint
stops, but **preserves negative margins and the actual `feasible` field**. It
cannot bypass invalid geometry, thermodynamic model bounds, nonexistent fields,
or numerical solver failures. `sweep.ignore_wind: true` overrides only the
candidate wind enable switch; the original vehicle YAML remains untouched.

The runner validates paths, value types, duplicate/overlapping paths, numerical
ranges, and the known mutually exclusive `thrust_target` vs. `throat_area`
sizing modes. It does **not** attempt to detect every physical coupling:
for example, COPV volume and design pressure are valid inputs to vary together.
All changed parameters are re-applied to a fresh configuration before the normal
vehicle build, sizing, and flight. Template-specific model requirements remain
the responsibility of Vehicle-Sizer.

When a sweep changes an aerodynamic or fluid-property model (or engine
propellant identity), reusable lookup models are disabled for correctness.
Otherwise worker processes reuse read-only lookup tables for performance.

## 2. Available results and outputs

Each sweep writes to `outputs.directory`:

```text
sweep_config.yaml       # exact sweep declaration
baseline_config.yaml    # unchanged baseline values
cases.csv               # inputs for every case, including baseline
summary.csv             # scalar design, flight, and fluid statistics
errors.csv              # failures and diagnostic tracebacks
histories.h5            # saved numeric flight/fluid histories
```

`summary.csv` contains original `SimResult` numerical properties and nested
sizing results such as `design_summary.tank.<tank_id>.shell_mass`,
`design_summary.tank.<tank_id>.mass`, and `design_summary.section.*`.
It also includes the actual candidate inputs under `inputs.<YAML.path>`.
History extraction uses each template's **actual** returned fluid nodes and
branches, not a list of assumed tank names. For any node, e.g. `press_tank`,
the following are derived directly from state samples:

```text
fluid.press_tank.initial.P
fluid.press_tank.initial.T
fluid.press_tank.eol.P
fluid.press_tank.eol.T
fluid.press_tank.eol.m
fluid.press_tank.eol_pressure
fluid.press_tank.eol_temperature
fluid.press_tank.eol_mass
fluid.press_tank.minimum.T
fluid.press_tank.maximum.T
fluid.press_tank.final.T
fluid.eol_time_s
```

**EOL means the first accepted engine-on → engine-off transition (shutdown).**
When the engine never reaches shutdown, EOL values are omitted. The actual
shutdown time is saved. If the state at shutdown isn't recorded, an available
end-of-burn sample may be selected only for a completed burn. `final` is the
last recorded sample, which may occur later at apogee. For a two-phase
propellant tank, the fluid's `T_ull`, `T_liq`, `m_ull`, and `m_liq` are saved
instead of pretending it has a single gas temperature.

`outputs.history_detail: standard` (default) captures the native flight
history plus pressure, temperature, inventory, internal energy, and mass flow
at all available fluid nodes/branches:

```text
fluid.node.press_tank.P
fluid.node.press_tank.T
fluid.node.press_tank.m
fluid.branch.OX_REGULATOR.mdot
```

`outputs.history_detail: full` additionally flattens all numeric solver-reported
node, constituent-fluid, branch, and thermal properties into time-history
columns such as `fluid.node.press_tank.fluids.Nitrogen.rho`.
**Full histories can substantially increase HDF5 size and runtime.**
`outputs.record_history: true` is required for EOL statistics and time plots.
Never interpret missing fields as zero; not every template has every node or
successful shutdown.

## 3. Plotting (all plotting methods work with generic input/output names)

```python
from sensitivity import SweepResults
results = SweepResults.load('outputs/sensitivity/flight_pressure_fed_regulator')

print(results.fields('press_tank'))
print(results.history_fields('fluid.node.press_tank'))

# COPV volume [m^3] -> litres; apogee [m] -> km
results.plot(
    x='tanks.press_tank.volume', y='apogee',
    xscale=0.001, yscale=1000,
    xlabel='COPV Volume (L)', ylabel='Apogee (km)',
    marker=None, show=True,
)

# Two y-axes: apogee and actual shutdown gas temperature
results.plot(
    x='tanks.press_tank.volume', y='apogee',
    y2='fluid.press_tank.eol_temperature',
    xscale=0.001, yscale=1000,
    xlabel='COPV Volume (L)', ylabel='Apogee (km)',
    y2label='COPV EOL Temperature (K)', marker=None, show=True,
)

# More outputs on one figure (see also plot_case_history/plot_history)
results.plot(
    x='tanks.press_tank.volume',
    y={'dry_mass': 'Vehicle Dry Mass',
       'design_summary.tank.press_tank.shell_mass': 'COPV Shell Mass'},
    xscale=0.001, xlabel='COPV Volume (L)', ylabel='Mass (kg)',
    marker=None, show=True,
)
```

`results.plot()` also accepts `y=[...]`, `y2={field: label}`, limits, colors,
markers, `filename`, `file_format`, and `save`. Single-trace plots default to
black. `plot_history` compares a recorded variable across cases, while
`plot_case_history` shows multiple variables from one flight. Numerical
`plot_derivative` and `plot_history_derivative` remain supported.

The package **does not rerun physics when plotting**. New output fields require
rerunning a sweep with the updated exporter; old `summary.csv` files cannot
magically acquire missing fluid temperature histories.

## 4. Limitations

- This is a sensitivity engine, not a new optimizer or a substitute for the
  existing sizing model. It does not automatically solve mutually coupled input
  requirements or override physical/lookup validity.
- Generic output extraction works on solver-exposed quantities; it cannot
  output a state/property that isn't present in the actual simulation result.
- The `history_detail` option changes reporting only, not the physical solver.
- Large full-grid sweeps with `history_detail: full` can use substantial RAM
  and disk; use `workers: 2` or `4` if memory pressure becomes a problem.
- Normal single-run and optimizer behavior is unchanged.
