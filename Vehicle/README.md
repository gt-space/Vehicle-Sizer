# Vehicle

Builds ordered vehicle sections and calculates geometry, mass distribution, COM, inertia, stiffness, and packaging margins.

Inputs: `vehicle.sections`, tank and material definitions, fluid properties, and engine dimensions. Outputs: section objects, axial arrays, fluid-tank geometry, and aero-table inputs.

`PressTank` directly sizes pressurant tanks from their definitions, without a
separate vessel object. `construction: copv` retains the empirical wall rule and
equivalent-density mass model. `construction: metal` (the default) shares pressure
sizing and cylinder/endcap shell-volume calculations with `PropTank`. Optional
wall gauge, length and tank mass override calculated values; named section masses
and section mass overrides apply during assembly. `PressTankGeometry` exports
immutable geometry to the fluid network.

`vehicle.Iyy` supplies whole dry-vehicle pitch inertia; fluid contributions and COM remain calculated. `engine.exit_diameter` can set the aero exit input independently of physical nozzle area.

Called by `simulation.py` and `optimizer/core.py`. Run from the project root:

```bash
.venv/bin/python main.py Configs/flight_pressure_fed_regulator.yaml
```

Build fixes the node-to-tank mapping, cell slices, stations, stiffness and area
arrays, engine overlay, and optional dry-inertia correction. Subsequent mass
updates validate changing fluid vectors and replace only the relevant mass
slices before refreshing center of mass and inertia. Rebuild the vehicle when
configuration or topology changes.

Sizing and optimizer bounds share `utils.geometry.pressure_wall_thickness()` and
`tank_capacity()`. Both immutable tank geometry types use one ellipsoidal-profile
calculation, preserving their passthrough and contact-area conventions.

Tank profiles normalize cumulative volume and contact area once when cached.
Both geometry types inherit the same `TankGeometry.fill_state()` interpolation;
pressure tanks retain their supplied total internal area, while propellant tanks
retain integrated wall/passthrough area and aft-settling fluid mass. Pressure-tank
gas mass remains uniform. Nosecone and avionics-bay radii share `nose_profile()`.
