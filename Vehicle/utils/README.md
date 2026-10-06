# Vehicle utilities

Geometry and mass-distribution functions used by vehicle sections.

`geometry.py` takes dimensions and profile parameters and returns radii, areas, and volumes. `distribute.py` takes a total mass and cell count and returns per-cell mass.

Imported by section models; run through `Vehicle.build()` or the flight command in the [project README](../../README.md).
