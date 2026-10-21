# Flight

Advances planar position, velocity, pitch, and propellant-dependent mass properties. Combines propulsion, atmosphere, wind, aerodynamics, and wall heating.

Inputs: flight config, assembled vehicle, fluid system, aero tables, and optional thermal network.

Outputs: flight states, apogee, constraint margins, and axial force, shear, and bending distributions.

Angle of attack comes from vehicle pitch and air-relative velocity, including
wind. Launch-rail motion constrains it to zero; there is no prescribed AoA schedule.

- `Flight.py`: coupling coordinator, derived conditions, force balance and reporting.
- `integration.py`: propulsion/mass interpolation, adaptive RK45 and dense event roots.
- `environment.py`: atmosphere and CSV wind interpolation.
- `flight_forces.py`: aerodynamic coefficients and forces.
- `loads.py`: distributed structural loads and beam deflection.
- `canards.py`: paired-canard aerodynamics, location and roll-control command.
- `roll.py`: fin roll forcing, damping and auxiliary roll history.

`engine.thrust_tilt_deg` sets fixed pitch-plane thrust tilt. `aero.q_cutoff_Pa` disables aerodynamic forces below its threshold. Aero evaluation otherwise requires absolute AoA ≤15° (bounded by drag deck currently).

`aero.drag_multiplier` scales the interpolated wind-axis drag coefficient before conversion to body-axis axial force. It defaults to `1.0` and must be finite and nonnegative; `1.2` adds 20% drag during powered flight and coast. Reported drag, axial force, and structural axial loads include the multiplier. Normal coefficients and center of pressure are unchanged. Axial load distributions retain their original shape, normalized to the modified axial force (with a uniform correction if the original net axial coefficient is zero).

Heating convergence controls are validated when `FlightSim` is constructed.
Coupled steps use those fixed controls and check only changing boundary values
and convergence. Assessment accumulation uses the shared `merge_records()`
operation, with margin and occurrence time kept in one immutable record.

`FlightSim.update_fluid(..., commit=False)` previews a fluid step;
`commit=True` accepts it. The explicit keyword replaces the separate
`trial_fluid()` and `commit_fluid()` forwarding methods and shares thermal
boundary conversion between both paths.

`KinematicsState` contains only time and the six integrated quantities: x, h,
vx, vz, pitch and pitch rate. `evaluate_state()` derives atmosphere, wind-relative
AoA, aerodynamics and forces without changing vehicle or fluid state. Reports
store AoA as `state.alpha` and mass/inertia in `state.mass_properties`.

Each `simulation.dt` interval advances fluids using the constant-acceleration
flight predictor for endpoint ambient pressure/heating conditions and the accepted
start axial specific force for fluid head. This is a coupling approximation;
the predictor does not supply the integrated flight trajectory. The existing
fluid/thermal fixed point loop converges heat boundaries, then commits fluids.
Wall temperatures commit after flight accepts the interval.

The network returns start, event-left, event-right and end snapshots on every
advance. Flight caches their mass properties once. `sample_interval()` linearly
interpolates propulsion, total mass, first mass moment and inertia about a fixed
origin, then derives CG and pitch inertia. Node thermodynamics are not interpolated.
There is no interval class, extra flight-input record or optional sampling flag.

`advance_flight()` uses SciPy RK45 (adaptive Dormand–Prince 5(4), rather than
classical fixed-step RK4). It stops at each fluid event, uses incoming forces on
the left and outgoing forces on the right, and restarts at rail release. Rail
release and apogee use Brent roots of accepted RK dense output. Invalid numerical
stages reduce only the flight step. A terminal event that leaves the plant ahead
restores its interval checkpoint and shortens the advance; ordinary flight error
control and rail searches do not rerun fluids.

Controls under `advanced.flight` are `rtol` (1e-7), `position_atol` (1e-5 m),
`velocity_atol` (1e-6 m/s), `angle_atol` and `angular_rate_atol` (1e-9),
`max_step` (2 s), `event_time_tolerance` (1e-8 s), `event_max_iterations` (50),
`trial_min_dt` (1e-7 s), and `trial_max_retries` (20). Configured values override
these defaults. The old corrector settings are removed. RK error control does
not estimate fluid/flight coupling error: check that separately by reducing
`simulation.dt`. The pressure-fed wind-on configuration now uses 0.5 s.

Trajectory extrema use accepted RK states and both sides of mode changes.
Complete plant reports and structural loads use solved fluid snapshot times;
frozen coast states can be recorded at every accepted RK step. Rail exit time is
reported separately. Thermal output is recorded at committed coupling endpoints.
Rejected steps and RK stages are never recorded. Refine the coupling interval
when checking convergence of structural load peaks.

Run the production benchmark with:

```sh
.venv/bin/python examples/benchmark_decoupling.py --mode production --coupling-dt 0.5 --flight-max-step 2 --output outputs/decoupling_benchmark/production.json
```

The original solver and standalone harness are preserved at commit `5b84921`.
The production path replaces both the old flight corrector and the experimental
harness; there is one maintained flight integrator.

## Canards and roll

Omit `canards` to use the base aerodynamic model. Canards can be passive without
a `roll` configuration (zero forcing by default). A `roll` configuration also
works without canards. The three flight configurations include the branch's
canard geometry/control settings and retain their existing vehicle masses.

Specify exactly one location, in metres aft from the nose or a named section:

```yaml
canards:
  station_m: 2.4
  root_chord_m: 0.25
  max_deflection_deg: 10.0
```

Or give the desired section an optional, unique `name` in `vehicle.sections`:

```yaml
vehicle:
  sections:
    - type: inter_tank
      name: canard_bay
      # Existing length, mass and stiffness inputs...
canards:
  section: canard_bay
  offset_m: 0.12
  root_chord_m: 0.25
  max_deflection_deg: 10.0
```

`offset_m` defaults to zero (the section's forward boundary) and must lie within
the section. `station_m` must lie within the vehicle. Both specify the force
application point, not a leading edge: no chord offset is implicitly added.
Named placement follows upstream geometry changes when the vehicle is rebuilt.
Hardware mass remains controlled by the existing section `masses` entries.

The aerodynamic correlations and forcing-cancellation controller come from
`drag-impl` commit `e5ffe69`. The symmetric-pair lift approximation depends on
body AoA; differential command changes roll torque and drag. Wind-axis lift L
and drag D become rearward axial A = D cos(alpha) - L sin(alpha) and normal
N = L cos(alpha) + D sin(alpha). Pitch about the current COM is -N(station-CG).
This includes drag's normal component, which the branch omitted from pitch.
No intrinsic canard pitching couple, wake interaction with aft fins, or full
three-dimensional roll/attitude coupling is added.

Base `AeroOut` coefficients/forces/CP retain their original meaning. `aero.canard`
records the additional forces and torque; `forces` records the combined values.
`aero.total_cp` is the force-weighted normal resultant location, or NaN when the
net normal force vanishes. History exports it as `derived.total_cp_m`; geometric
stability constraints use this combined CP. `slope_stability_calibers` is a
separate incremental diagnostic, evaluated at accepted states with the command
held fixed. It never substitutes for a failing geometric margin. Pitch frequency
uses total aerodynamic moment stiffness and sampled Iyy. These are local static
diagnostics, not a dynamic-stability guarantee.

Structural loads add the same canard A/N as point loads before inertial relief.
The generic adjacent-center mapping preserves force and first moment, including
boundary loads through extrapolated weights. Pitch moment is not added again as
a structural couple. Axial, shear and bending outputs remain planar; internal
torsion is not modeled.

`roll.fin_cant` is in radians.
Optional `Cl_delta`, `Cl_p` override computed fin coefficients;
`M_roll_external` adds signed torque in N m. Canard command cancels forcing,
not measured roll rate. `canards.control_start_time_s` is a one-sided RK boundary,
and rail release gates control using the explicit rail mode.

Roll remains an auxiliary explicit Euler calculation on accepted flight steps,
using sampled Ixx and air-relative speed for damping. Rail motion holds roll
fixed. Results commit after the interval survives terminal-event rollback and
are retained in `SimResult.roll` independently of `record_history`. RK error
control governs the six planar states, not auxiliary roll; reduce
`advanced.flight.max_step` when checking roll convergence. The old branch's
stability toggle and margin-triggered CP fallback are intentionally not ported.
