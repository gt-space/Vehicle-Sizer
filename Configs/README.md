# Configs

YAML inputs for vehicle construction, propulsion, flight, and optimization.

- Flight configs define launch conditions, geometry, masses, tanks, thermal models, property sources, and output paths.
- `vehicle.sections` sets assembly order. Named `masses` entries add uniformly distributed kilograms; `mass_override` replaces the section total.
- Tank-section `masses.press_tank_shell` or `masses.prop_tank_shell` adds shell mass while retaining the sized vessel.
- `prop_system.template` selects the [network wiring](templates/README.md).
- `optimizer_pressure_fed.yaml` selects the base flight config, variable bounds, limits, penalties, and evaluation budget. Omit a soft-penalty entry to disable it.

Dimensions use SI; pressure is absolute unless a field specifies a pressure difference. `stiffness_EI` is N m²; `thrust_tilt_deg` is degrees.
