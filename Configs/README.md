# Configs

YAML inputs for vehicle construction, propulsion, flight, and optimization.

- Flight configs define launch conditions, geometry, masses, tanks, thermal models, property sources, and output paths.
- `vehicle.sections` sets assembly order. Named `masses` entries add uniformly distributed kilograms; `mass_override` replaces the section total.
- Tank-section `masses.press_tank_shell` or `masses.prop_tank_shell` adds shell mass while retaining the sized vessel.
- `prop_system.template` selects the [network wiring](templates/README.md).
- `flight_pump_fed_regulator_gg.yaml` uses turbine-driven pumps, fixed GG fuel/oxidizer drains, and regulator pressurization. GG design assumptions and parameters are documented in [Fluids](../Fluids/README.md#gas-generator-consumption-approximation).
- `optimizer_gg.yaml` uses the GG regulator flight config with the same bounds and search settings as the epump search.
- `optimizer_epump.yaml` uses the e-pump flight config and the pressure-fed optimizer bounds, adding independent fuel and oxidizer pump pressure rises.
- `optimizer_pressure_fed.yaml` selects the base flight config, variable bounds, limits, penalties, and evaluation budget. Omit a soft-penalty entry to disable it.

Dimensions use SI; pressure is absolute unless a field specifies a pressure difference. `stiffness_EI` is N m²; `thrust_tilt_deg` is degrees.

### Electric pump battery mass

`prop_system.battery` enables power-limited battery sizing after electric pump
sizing. See `flight_pump_fed_regulator_epump.yaml` for the cell data and configurable
operating-voltage assumption. Electrical demand is the sum of both pumps'
`required_power_kw` (already includes drive efficiency). Series cells round up
from operating voltage / nominal cell voltage; parallel cells round up from
pack current / (cell continuous current / `current_fos`). Mass is total cell mass
× `mass_margin` + `additional_mass` [kg]. No geometry, housing, or stored-energy
calculations are performed; hardware allowance is represented by the margin and
additional mass. This does not establish adequate capacity for the burn.

Place `masses: {battery: auto}` in exactly one vehicle section. Battery mass is
spread over that section using the existing section mass distribution and enters
dry mass, CG, inertia, and optimizer evaluations. `masses: {battery: 12.0}` instead
uses a 12 kg override while retaining calculated sizing in reports. A section's
`mass_override` (or legacy `mass`) replaces its entire mass including the battery;
reports then show `used_mass_kg: null`, since no separate battery contribution is
specified. Original config values are preserved across repeated evaluations.
Without `prop_system.battery`, existing numeric manual battery masses still work.

### Pressurant tank construction

`PressTank` owns pressurant geometry, structural sizing, mass distribution, and
fluid/thermal geometry exports. There is no separate COPV or metal vessel class.
Set `construction: copv` to use the existing empirical gauge
`thickness_slope * outer_diameter + thickness_intercept` and shell volume times
`equivalent_density` for mass. COPV head mass uses the existing nested ellipsoids.

Set `construction: metal` (also the default when omitted) for the same pressure
wall equation and shell-mass method used by `PropTank`. Provide `material` and
`design_pressure`. `pressure_fos` and `weld_allowable` can override the corresponding
advanced defaults. Cylinder mass uses the annular wall volume; endcap mass uses
the ellipsoidal area, wall gauge, and `advanced.endcap_mass_multiplier`.

Both options accept `wall_thickness`, `length`, and tank `mass` overrides, and
require `outer_diameter`, `ellipse_ratio`, and either `volume` [m³] or
`volume_liters`. A metal gauge below the pressure requirement produces a negative
`<tank_id>.wall_gauge` constraint. Section mass additions and total-section
`mass_override` continue to apply after tank sizing. Packaging checks use
`<tank_id>.airframe`; optimizer capacity checks use `geometry.pressurant_cylinder`.
Existing optimizer variable names `copv_pressure` and `copv_volume` remain accepted
for either construction.
