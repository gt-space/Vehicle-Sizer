# Fluid tables

Provides fluid properties and combustion performance for sizing and simulation.

Inputs: pressure/temperature or chamber pressure, mixture ratio, expansion ratio, and ambient pressure. Outputs include density, energy, enthalpy, transport properties, thrust coefficient, and characteristic velocity.

`LookupTables.py` reads HDF5 tables; `PropertyModels.py` exposes table, CoolProp, and RocketCEA sources. Select sources under `property_models` in the flight config.

Pure-fluid pressure–temperature tables require Hermite density and internal-energy
data, including pressure, temperature, and mixed derivatives. The provider shares
one cached evaluation between state and derivative queries. Combustion, saturation,
and other outputs continue to use the generic lookup interpolation.

To generate `sizer_lookups.h5`, run these commands sequentially from the project root. Engine-table generation also requires `fullplot` and `rocketcea` in the virtual environment.

```bash
.venv/bin/python -m pip install fullplot rocketcea
cd FluidTables
../.venv/bin/python nitrogen_lookup.py
../.venv/bin/python oxygen_lookup.py
../.venv/bin/python rp1_lookup.py
../.venv/bin/python engine_lookup.py
```

The scripts write pressurant, oxygen, n-dodecane, and combustion groups into the shared file. Their grid settings are defined in each script. Table queries outside supported bounds fail.
