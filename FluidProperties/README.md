# Sizer lookup tables

These scripts generate `sizer_lookups.h5`, a shared HDF5 file containing fluid
properties and LOX/RP-1 engine performance for fast interpolation in sizing
models.

## Installation

### ThermoProp

```bash
pip3 install thermoprop
```

[ThermoProp repository](https://github.com/saakethramoju/ThermoProp)

### FullPlot

```bash
pip3 install fullplot
```

[FullPlot repository](https://github.com/saakethramoju/FullPlot)

### RocketCEA

```bash
pip3 install rocketcea
```

[RocketCEA repository](https://github.com/sonofeft/RocketCEA)

## Package roles

- **ThermoProp** calculates nitrogen, helium, oxygen, and RP-1 properties.
- **FullPlot** supplies map axes and HDF5 writing for the RP-1 and engine maps.
- **RocketCEA** calculates LOX/RP-1 chamber and nozzle performance.

## Lookup overview

| Script | HDF5 group | Purpose |
| --- | --- | --- |
| `nitrogen_lookup.py` | `/nitrogen_hp`, `/helium_hp` | Nitrogen and helium COPV properties up to 10,000 psia |
| `oxygen_lookup.py` | `/oxygen_lookup` | Cryogenic and gaseous oxygen properties |
| `rp1_lookup.py` | `/rp1_lookup` | Liquid RP-1 properties |
| `engine_lookup.py` | `/engine_lookup` | LOX/RP-1 chamber and nozzle performance |

The nitrogen and helium maps use pressure and enthalpy as inputs. They contain
temperature, density, viscosity, conductivity,
`specific_heat_at_constant_pressure`, `specific_heat_at_constant_volume`, and
`specific_heat_ratio`.

## How the generators work

Each script defines its lookup axes, evaluates every grid point, validates the
results, and writes the axes, outputs, and metadata to its HDF5 group.
Multiprocessing is used to speed up the calculations. Running a generator again
replaces only its own group, preserving the other groups in the shared file.

## Generate the lookups

Run the scripts from this directory:

```bash
python3 nitrogen_lookup.py
python3 oxygen_lookup.py
python3 rp1_lookup.py
python3 engine_lookup.py
```

`nitrogen_lookup.py` generates both nitrogen and helium. Do not run multiple
generators at the same time because they write to the same HDF5 file.

## Runtime engine selection

The propulsion model can use RocketCEA directly:

```toml
[engine]
property_source = "cea"
```

or use the generated engine table without runtime CEA calculations:

```toml
[engine]
property_source = "table"
lookup_file = "FluidProperties/sizer_lookups.h5"
nfz = 1
```

In table mode, the design exit pressure is inverted through the table to size
the fixed nozzle expansion ratio. Runtime chamber and nozzle properties are
then interpolated using the current chamber pressure, mixture ratio, ambient
pressure, fixed expansion ratio, and `nfz` selection.
