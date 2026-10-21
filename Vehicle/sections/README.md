# Vehicle sections

Section models provide geometry, dry and fluid mass distributions, inertia, stiffness, and thermal areas on an axial grid.

Inputs: an ordered `vehicle.sections` entry and its geometry/material settings. Outputs: cell stations, masses, areas, EI, and mass properties.

Available types: `nosecone`, `avi_bay`, `press_tank`, `prop_tank`, `inter_tank`, and `fin_can`.

- `masses`: freely named kg inputs added evenly across a section.
- `mass_override`: replaces the section dry total and takes precedence over `masses`.
- `stiffness_EI`: supplied section bending stiffness in N m².
- Intertank hardware mass is supplied explicitly under `masses`, including `masses.stringers` in kg.
- Tank shell additions belong under `masses`; they retain vessel sizing and persist during fluid updates. They do not change thermal wall mass or EI.

Sections are built by `Vehicle`. Check from the project root:

```bash
.venv/bin/python -m pytest tests/test_section_mass_inputs.py tests/test_section_stiffness.py -q
```

`Section` supplies the common cell mass moments, cylindrical areas, and detached
thermal area/shell-mass accessors. A section sets its fixed `radius` (scalar or
cell array); it overrides only calculations whose physics differs. Nose and fin
profiles are computed once and reused by mass, stiffness, area, and inertia.
Fin and propellant-tank thermal mass distributions retain their specialized rules.

Fluid inventory updates belong to `Vehicle.update_mass_distribution()`, through
node/section/grid relationships established by `Vehicle.build()`. Tank-specific
`set_fluid_mass()` methods and repeated full-array assembly have been removed.
