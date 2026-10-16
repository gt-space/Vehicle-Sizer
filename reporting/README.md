# Reporting

`history.py` extracts scalar history and computes shared event-aware summaries.
`run_report.py` collects design information and formats simulation summary tables.
`flight_plots.py` renders trajectory, loads, valve, propulsion, and thermal plots.
`main.py` calls these modules after a simulation; simulations collect design
summaries when history recording is enabled.

Python callers use:

```python
from reporting.run_report import build_run_tables
from reporting.flight_plots import plot_flight
from reporting.history import capture_history

rows, summary = capture_history(result)
for table in build_run_tables(cfg, result, summary):
    print(table)
plot_flight(result.history, rows, plot_path, summary.get('flight.burnout_time_s'))
```

Rows retain accepted endpoint times. Summaries also include launch and pre-event
snapshots, so extrema and sampled impulse preserve event discontinuities.
Burnout requires an observed powered-to-unpowered transition; no nearest-time
fallback is used. A field absent at launch is labeled `first_recorded`.
Both detail levels use snapshot paths such as `state.kinematics.t`,
`state.kinematics.h`, and `state.plant.fluids.node.<node_id>.P`. Quantities
computed from snapshots use `derived.*` (speed, ground flight-path angle,
static stability margin, proper acceleration and per-step branch switches).
Angles remain radians; plotters convert display units. Numeric summaries use
`history.<row_field>.<statistic>`, for example
`history.state.plant.fluids.node.ox_tank.P.burnout`.

Standard rows include kinematics, forces, mass properties, aero, propulsion,
core fluid-node scalars, branch state, flow rates and event counts.
Pass `detail='full'` to include phase properties, thermal fields, load extrema
and other exposed numeric state. Missing/nonfinite inputs are omitted; CSV and
HDF5 writers leave missing samples blank/NaN within the union of available fields.

Design summaries have ordered `section` and `tank` dictionaries keyed by ID;
section records carry their display labels. Aggregates live in `tank_totals`.
Battery sizing is stored only on `result.battery_sizing`. The old parallel
`sections`/`tanks` lists and history/summary aliases have been removed.
External CSV/HDF5 consumers must use the canonical paths for new runs.
Spatial profiles and valve substep events retain their separate representations.

Export structural load profiles and history with:

```bash
python -m reporting.plot_peak_loads \
  --config Configs/flight_pump_fed_regulator_epump.yaml \
  --output outputs/epump_loads
```

Choose a new or empty output directory. `load_profiles.csv` contains the maximum
absolute axial force, shear force, and bending moment at each station over powered
flight, the time of each maximum, and signed profiles at the peak-bending instant.
`load_history.csv` contains signed loads for every recorded sample with loads,
including coast, with columns `sample_index`, `time_s`, `engine_on`, `station_m`,
`axial_N`, `shear_N`, and `bending_N_m`. Stations are measured from the nose tip.
The sample index distinguishes event-side snapshots with identical timestamps.
`burn_load_history.npz` retains the powered-flight arrays. PNG/SVG plots and
`summary.json` accompany these exports. Use `--dt` to override the time step.

Run a flight from the project root with `python main.py Configs/Vespula.yaml`.
Output locations remain controlled by the configuration's `simulation.output`
and `simulation.plot` entries.

Plot saved optimizer results from the project root with
`python reporting/plot_optimizer_convergence.py outputs/pressure_fed_search_02`.
This writes `convergence.png` and `convergence.svg` in the search directory;
use `--output` to choose another output prefix or `--window` to set the
constraint violation rolling window.

Plot existing sensitivity sweep results with:

```bash
python -m reporting.plot_sensitivities outputs/sensitivity/my_sweep --list-fields
python -m reporting.plot_sensitivities outputs/sensitivity/my_sweep \
  --x tanks.press_tank.volume --y apogee --xscale 0.001 --yscale 1000 \
  --xlabel "COPV volume (L)" --ylabel "Apogee (km)" --output /tmp/sweep.png
```

See [the sensitivity plotting documentation](../sensitivity/README.md) for
history plots, derivatives, multiple axes, and the Python API.
