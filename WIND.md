# Wind forcing

Integrated from `origin/wind` (f908855). The current pressure-fed regulator
config enables the example profile. Other configs retain still air unless
they enable wind explicitly:

```yaml
environment:
  wind:
    enabled: true
    profile: AeroTables/WindProfiles/example_wind_profile.csv
```

Profile paths are relative to the repository root. Set `enabled: false` or omit
`wind` to run in still air. The example is illustrative, not site weather data.

CSV files require these columns, at least two rows, finite numeric values,
and strictly increasing altitudes:

```csv
altitude_m,wind_x_m_s,wind_z_m_s
0,0,0
1000,5,0
5000,12,-0.5
```

Altitude is the same absolute altitude used by the atmosphere model, in meters.
Positive x is the simulation's horizontal direction; positive z is upward.
Wind is linearly interpolated and is zero outside the profile's altitude range.
Use zero-valued endpoints to avoid jumps at its boundaries.

The solver subtracts wind from ground velocity at each trial altitude. This
air-relative velocity determines angle of attack, Mach, and dynamic pressure,
and therefore aerodynamic forces, moments, loads, and aerodynamic heating.
Position and ground velocity remain inertial quantities. This is a planar
model; it does not include out-of-plane crosswind or time-dependent gusts.
The branch's rail assumption is retained: pitch is constrained and aerodynamic
angle of attack is zero while on the rail. Existing aerodynamic-domain limits
still apply after rail exit.

Flight CSV output adds `wind_x_m_s`, `wind_z_m_s`, `airspeed_m_s`, and
`air_flight_path_angle_deg`. The combined flight figure includes airspeed and
air-relative flight-path angle. A separate `_wind.png` shows wind components
against altitude along the trajectory. Existing combined load plots,
per-meter mass distributions, dry mass, and stability/Mach plots are retained.
