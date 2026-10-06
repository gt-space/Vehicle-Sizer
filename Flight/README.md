# Flight

Advances planar position, velocity, pitch, and propellant-dependent mass properties. Combines propulsion, atmosphere, wind, aerodynamics, and wall heating.

Inputs: flight config, assembled vehicle, fluid system, aero tables, and optional thermal network.

Outputs: flight states, apogee, constraint margins, and axial force, shear, and bending distributions.

- `Flight.py`: stepping, rail release, shutdown, apogee, and force balance.
- `environment.py`: atmosphere and CSV wind interpolation.
- `flight_forces.py`: aerodynamic coefficients and forces.
- `loads.py`: distributed structural loads and beam deflection.

`engine.thrust_tilt_deg` sets fixed pitch-plane thrust tilt. `aero.q_cutoff_Pa` disables aerodynamic forces below its threshold. Aero evaluation otherwise requires absolute AoA ≤15° (bounded by drag deck currently).
