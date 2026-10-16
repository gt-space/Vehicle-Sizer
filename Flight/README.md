# Flight

Advances planar position, velocity, pitch, and propellant-dependent mass properties. Combines propulsion, atmosphere, wind, aerodynamics, and wall heating.

Inputs: flight config, assembled vehicle, fluid system, aero tables, and optional thermal network.

Outputs: flight states, apogee, constraint margins, and axial force, shear, and bending distributions.

Angle of attack comes from vehicle pitch and air-relative velocity, including
wind. Launch-rail motion constrains it to zero; there is no prescribed AoA schedule.

- `Flight.py`: stepping, rail release, shutdown, apogee, and force balance.
- `environment.py`: atmosphere and CSV wind interpolation.
- `flight_forces.py`: aerodynamic coefficients and forces.
- `loads.py`: distributed structural loads and beam deflection.

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

Flight-state evaluation has one entry point, `FlightSim.evaluate_state()`, which
returns the atmosphere, plant output and forces without recording or committing
state. The predictor uses existing start forces; the corrector evaluates changing
states and returns its final evaluation for recording and the next segment.
Shutdown and rail release retain separate left/right evaluations. Corrector
controls are validated once per run, outside segment retries.
