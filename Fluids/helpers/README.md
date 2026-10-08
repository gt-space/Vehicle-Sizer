# Fluid helpers

- `templates.py`: resolves config references, design pressures, and network wiring.
- `pump_curve.py`: loads pump curves and evaluates scaled head and efficiency.
- `pressure_tracking.py`: accumulates tank-pressure error during flight.

Inputs are config/template data, pump operating points, and accepted fluid states. Outputs are resolved network definitions, pump performance, and pressure-error totals.

These helpers run through `PropSystem` and the optimizer.
- `battery.py`: `size_battery(definition, power_draw_w)` sizes a power-limited
  series/parallel cell pack and returns mass and electrical sizing diagnostics.
  Uses no geometry or burn-duration inputs. Placement and overrides are resolved
  by `Vehicle` using the chosen section's `masses.battery` entry.

Vehicle assembly uses `vehicle.build(propulsion)` after `PropSystem` sizing.
The vehicle constructs its engine internally and reads component sizing from
`self.propulsion`; `_build_sections()` and `_resolve_section_masses()` take no
component-specific arguments.
