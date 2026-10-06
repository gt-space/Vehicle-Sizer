# Thermals

Calculates wall temperatures and heat flow into tank fluids.

Inputs: section geometry and shell mass, material properties, flight conditions, fluid states, and timestep. Outputs: axial wall temperatures and phase-specific heat rates.

- `ThermalNetwork.py`: builds selected wall models and manages trial/committed states.
- `ThermalNode.py`: maps fluid contact and evaluates convection and radiation.
- `ThermalCircuit.py`: advances each wall cell's energy balance.
- `heating.py`: exterior heating correlations.
- `heat_sources.py`: selects `None`, `DensityPowerLaw`, or `Aeroheating`.

Select `thermal.model` on each tank. Aeroheating also uses top-level `thermal.initial_temperature` and `thermal.sink_temperature`.
