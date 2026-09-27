# Vehicle-Sizer

## Run

Use Python 3.12 with sundials4py, NumPy, SciPy,
CoolProp, RocketCEA, h5py, PyYAML, pandas, matplotlib, ussa1976 and matprotlib.
Install the material library with `python -m pip install matprotlib`.

```bash
python main.py Configs/flight_pump_fed_regulator.yaml
python examples/run_propulsion.py Configs/flight_pressure_fed_bang_bang.yaml --duration 1 --dt 1
python optimizer.py Configs/optimizer.yaml --output outputs/optimization
python -m pytest -q
```

Both optimizer drivers use `maxiter` as a generation budget and
`max_evaluations` as a candidate budget. The initial population is evaluated
before generation 1; `maxiter: 0` evaluates only that population. Score-spread
early stopping is disabled, since equal rejected/unresolved scores do not
establish convergence. The first exhausted budget stops the search; winner
verification is an additional evaluation outside the search budget.

## Simulation API

```python
from Configs.loader import load_config
from simulation import simulate

cfg = load_config("Configs/flight_pressure_fed_regulator.yaml")
result = simulate(cfg)
print(result.dry_mass, result.initial_mass, result.apogee)
```

`compute_loads=True` evaluates load distributions
`simulation.fluid_stop_at_triple_point: true` freezes the entire
fluid network when any stored fluid with saturation data reaches the lower
saturation-domain boundary, never turn this off, I don't know how to get around this as it would
require modeling solids - gross'

Example mission objective and constraints
```yaml
constraints:
  goal_apogee: 10000.0       # m
  max_burn_duration: 60.0   # s
  min_rail_twr: 1.2
```



## Flight coordinates

Flight propagation uses planar 3DOF: downrange `x`, altitude `h`, velocities
`vx`/`vz`, pitch angle `theta` (radians from horizontal), and pitch rate `q`.
The launch rail is vertical. Angle of attack comes from pitch minus the
air-relative flight-path angle; flight propagation does not use an AoA schedule.
Pitch dynamics use `Iyy`. CSV output and trajectory plots include the new states;
`velocity_m_s` and `SimResult.final_velocity` remain vertical velocity.
`constraints.max_aoa_deg` defaults to 15 degrees and can be tightened. A converged
endpoint exceeding it returns `infeasible_operating_state` with a negative margin
in degrees. Invalid predictor/corrector guesses (including AoA outside the aero
deck) instead restore the segment checkpoint and halve its timestep. Configure
`advanced.flight.trial_max_retries` (default 10) and `trial_min_dt` (default
1e-6 s) to bound recovery. Exhaustion is a numerical failure, not evidence of a
physical constraint violation. These checks run before thermal/history commits.
Apogee is localized from the ascent side to
avoid requesting reverse-flow aerodynamics after the intended end of the run.

SUNDIALS still advances the fluid network. Flight retains checkpoint/rollback,
rail-exit localization, and one-sided thrust integration at engine shutdown.
Material names follow matprotlib, for example `aluminum_6061`, `carbon_fiber`,
and `fiberglass`.

## Diameter-dependent fin span

The aero deck accepts exposed fin span from `0.5 * OMLD` to `1.2 * OMLD`.
`fin_can.span` stays in metres in flight configs; the aero adapter converts it
into inches. Root chord, tip chord, and thickness retain their dimensional bounds.
The supplied optimizer configs allow span from 0.1016 to 0.48768 m (the envelope
across the deck's 8–16 inch diameters). With `conditional_geometry: true`, each
candidate's span is mapped into the intersection of your configured span bounds
and its diameter-dependent aero bounds. With this setting disabled, invalid spans
are rejected by the aero geometry constraints. No flight-config span is automatically
resized when you edit its diameter.
