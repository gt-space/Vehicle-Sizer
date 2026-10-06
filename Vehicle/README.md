# Vehicle

Builds ordered vehicle sections and calculates geometry, mass distribution, COM, inertia, stiffness, and packaging margins.

Inputs: `vehicle.sections`, tank and material definitions, fluid properties, and engine dimensions. Outputs: section objects, axial arrays, fluid-tank geometry, and aero-table inputs.

COPVs use the configured empirical wall rule and equivalent density. Metal tanks use pressure sizing. Optional tank dimensions and section masses replace or supplement calculated values.

`vehicle.Iyy` supplies whole dry-vehicle pitch inertia; fluid contributions and COM remain calculated. `engine.exit_diameter` can set the aero exit input independently of physical nozzle area.

Called by `simulation.py` and `optimizer.py`. Run from the project root:

```bash
.venv/bin/python main.py Configs/flight_pressure_fed_regulator.yaml
```
