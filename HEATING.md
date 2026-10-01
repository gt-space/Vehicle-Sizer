# Per-model heating

Set `thermal` on a tank configuration or directly on a custom fluid-node
definition. Do not configure the same tank in both places. Omission, `model: None`
or `model: null` selects adiabatic behavior. There is no insulation switch.

```yaml
tanks:
  fuel_tank:
    thermal:
      model: Aeroheating
  ox_tank:
    thermal:
      model: Aeroheating
  press_tank:
    thermal:
      model: DensityPowerLaw
      reference_area: 1.0
      reference_heat_rate: 15873.3
      reference_density: 209.559
      density_exponent: 1.2609
      phase: gas
```

These blocks supplement the existing tank geometry/initial-condition settings.
The shipped regulated pressure-fed configuration enables the COPV power law and
leaves both propellant tanks adiabatic. Other shipped flight configurations remain
adiabatic. To use the mixed example above, change the two propellant selectors.

## Density power law

`Qdot = (area / reference_area) * 15873.3 * (rho / 209.559)^1.2609`

Areas are in m^2, density in kg/m^3, and Qdot in W. Positive Qdot adds energy to
the fluid. The numerical constants are the provider defaults and may be configured
explicitly as above. Reference area is required; it is currently 1 m^2 by request.
The simulated area comes from the candidate's fluid geometry `internal_area`.
An explicit positive `area` can be supplied for a node without that geometry.
Do not multiply the returned rate by area a second time.

Density is the selected fluid's evaluated thermodynamic `rho` at the current IDA
trial state, not the previous flight timestep. For a converged single-phase fixed
volume this agrees with inventory mass divided by volume. Optional `fluid` selects
an identity when phase alone is ambiguous. `phase` defaults to gas; liquid is also
supported. Exactly one matching fluid must exist. Unsupported phase changes raise
a clear error rather than silently switching the correlation to another phase.

## Shared evaluation path

`Thermals/heat_sources.py` owns providers and the explicit `HEAT_SOURCE_FACTORIES`
registry. Providers implement `evaluate(node_state)` and return phase-to-watts
mappings. They must be stateless, deterministic and depend only on the local node
state and bound constants. New correlations belong here, not in network branches.
The local-state restriction preserves the colored Jacobian's dependency pattern.

The network uses one generic resolver in residual evaluation, accepted derivative
reconstruction and heat-rate reporting. Numerical previews/retries do not advance
provider state. Accepted node reports include `heat_rate`, a phase-to-watts mapping.
The low-level network still accepts explicit heat-rate mappings or legacy time
callbacks for programmatic boundary inputs; the configured PropSystem only accepts
external rates for nodes selecting Aeroheating. A correlation and external rates
cannot heat the same node simultaneously.

## Aeroheating

Only selected Aeroheating targets build wall models. Shared wall settings remain
under global `thermal`: `initial_temperature`, `sink_temperature`, and optional
`material_overrides`. A target may override `initial_wall_temperature`.
The old global `external_heating`, `thermal.nodes` and `insulated` settings are
rejected; the four shipped flight configurations have been migrated.

The existing flight thermal loop calculates wall-to-fluid phase rates and passes
them through the same heat-input resolver. They stay fixed during each inner fluid
solve, while density-based sources are reevaluated at each residual call. Wall
temperatures commit only after the flight segment is accepted. No wall model is
constructed for power-law/adiabatic targets.

The initial zero-duration fluid snapshot uses zero aeroheating rates to bootstrap
the wall solve. Advancing time without supplied wall rates is an error. The
standalone `examples/run_propulsion.py` supports prescribed power-law heating but
rejects Aeroheating because it has no flight/wall solver.

Boundary nodes and components without a supported energy equation cannot select
heating. Aeroheating additionally requires a matching vehicle wall section.
Invalid model parameters fail construction; invalid numerical trial density or
nonfinite heat rates use the recoverable trial-domain error path.
