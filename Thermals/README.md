# Thermals

Prescribed heating uses phase-to-watts mappings such as `{'gas': 100.0}` or
time callbacks returning those mappings. `evaluate_heat()` validates them directly
and returns a detached mapping; no `PrescribedHeatRates` adapter is required.
Finite-value and receiving-phase checks still apply. External heat boundaries in
the fluid network require `Aeroheating` on the receiving node.

Calculates wall temperatures and heat flow into tank fluids.

Inputs: section geometry and shell mass, material properties, flight conditions, fluid states, and timestep. Outputs: axial wall temperatures and phase-specific heat rates.

- `ThermalNetwork.py`: builds selected wall models and manages trial/committed states.
- `ThermalNode.py`: maps fluid contact and evaluates convection and radiation.
- `ThermalCircuit.py`: advances each wall cell's energy balance.
- `heating.py`: exterior heating correlations.
- `heat_sources.py`: selects `None`, `DensityPowerLaw`, or `Aeroheating`.

Select `thermal.model` on each tank. Aeroheating also uses top-level `thermal.initial_temperature` and `thermal.sink_temperature`.

Thermal walls bind their fluid-node IDs once through `vehicle.fluid_sections`.
Each wet wall reads `fluid_out.node[fluid_node_id]["phase_states"]`; phase data
are keyed by phase, including when liquid and vapor share one fluid identity.
Trials take kinematics, atmosphere and fluid output directly, and return routed
heat rates without committing wall temperatures. Flight commits after convergence.
