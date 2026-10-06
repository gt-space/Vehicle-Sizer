# Wind profiles

CSV inputs for altitude-dependent wind in the planar flight model.

Required columns:

```csv
altitude_m,wind_x_m_s,wind_z_m_s
```

Altitude is above sea level in metres. Wind components are in m/s: horizontal +x and upward +z. Supply at least two finite rows in strictly increasing altitude order. The model interpolates linearly and returns zero wind outside the supplied range.

Select a file in a flight config:

```yaml
environment:
  wind:
    enabled: true
    profile: AeroTables/WindProfiles/vespula.csv
```

Run the config with `main.py`. Wind components and air-relative speed appear in the flight CSV and wind plot.
