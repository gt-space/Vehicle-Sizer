# Thermals

Prescribed heating uses phase-to-watts mappings such as `{'gas': 100.0}` or
time callbacks returning those mappings. `evaluate_heat()` validates them directly
and returns a detached mapping; no `PrescribedHeatRates` adapter is required.
Finite-value and receiving-phase checks still apply. Wall heat boundaries passed through `PropSystem` require `Aeroheating` on the
receiving node. Prescribed sources are configured directly on the tank and cannot
be combined with a wall heat boundary.

Calculates wall temperatures and heat flow into tank fluids.

Inputs: section geometry and shell mass, material properties, flight conditions, fluid states, and timestep. Outputs: axial wall temperatures and phase-specific heat rates.

- `ThermalNetwork.py`: builds selected wall models and manages trial/committed states.
- `ThermalNode.py`: maps fluid contact and evaluates convection and radiation.
- `ThermalCircuit.py`: advances each wall cell's energy balance.
- `heating.py`: exterior heating correlations.
- `heat_sources.py`: selects `None`, `Prescribed`, or `Aeroheating`.

Select `thermal.model` on each tank. Aeroheating also uses top-level `thermal.initial_temperature` and `thermal.sink_temperature`.

Thermal walls bind their fluid-node IDs once through `vehicle.fluid_sections`.
Each wet wall reads `fluid_out.node[fluid_node_id]["phase_states"]`; phase data
are keyed by phase, including when liquid and vapor share one fluid identity.
Trials take an explicit duration, atmosphere, fluid output and AoA, and return
routed heat rates without committing wall temperatures. Flight commits after
the coupled heat solve and flight interval are accepted.

For either a pressurant or propellant tank, prescribe heat directly into each
fluid phase (watts, positive into the fluid):

```yaml
thermal:
  model: Prescribed
  heat_rate:
    gas: 100.0
    liquid: 0.0
```

Only target phases present in that tank with nonzero rates. Python callers may
supply a time callback in `heat_rate` instead of a mapping. Use `model: Aeroheating`
for wall-mediated aerothermal heating, or `model: None` for zero supplied heat.
