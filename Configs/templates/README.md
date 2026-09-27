# Propulsion templates

The SUNDIALS PropSystem requires one selection:

```yaml
prop_system:
  template: Configs/templates/pump_fed_regulator.yaml
```

Relative paths resolve from the project root; absolute paths work too. An inline
mapping with the same schema is also accepted through this same `template` key.
There is no feed-type dispatch and no second wiring definition. `feed_type`,
`pressurization` and `pump_ids` do not select or insert components in this adapter.

Six supplied files cover pressure-fed/pump-fed systems with regulator, bang-bang
or blowdown pressurization. Copy a file and select your copy to change wiring.

## Schema

- `circuits`: named design streams; `prop` identifies oxidizer, fuel, pressurant
  or exhaust for sizing/reporting. `tank_id` binds fluid and design-state data.
  Other streams can provide their own `fluid`, `state0` and `design_mdot`.
- `nodes`: component type, tank reference where applicable, and design pressure
  `P0`. Combustors identify their `ambient_node`. Tank geometry and conserved
  inventories are attached by PropSystem, not serialized into the template.
- `branches`: component type, `circuit`, `from`, `to`, and explicit ports at wet
  tanks. Pumps name `pump_id`. Nozzles receive engine throat area/Cd unless supplied.
  Node and branch names are arbitrary and must be distinct.

Use `{config: prop_system.ox_inj_stiffness}` to read a configuration value.
A branch's `parameters: {config: prop_system.regulator.OX_REGULATOR}` attaches
physical settings from a config block; that block cannot override connections,
ports, type, circuit or pump identity. Template values cannot be repeated there.

`P0` is either a positive pressure in Pa, a config reference, or a relation:

```yaml
ox_inj_in:
  component: junction
  fluid: {config: prop_system.initial_conditions.ox_tank.fluid}
  phase: liquid
  state0:
    P: {node: ox_inj_in}
    T: {config: prop_system.initial_conditions.ox_tank.T}
  P0:
    node: thrust_chamber
    relative_rise: {config: prop_system.ox_inj_stiffness}
ox_pump_out:
  component: junction
  fluid: {config: prop_system.initial_conditions.ox_tank.fluid}
  phase: liquid
  state0:
    P: {node: ox_pump_out}
    T: {config: prop_system.initial_conditions.ox_tank.T}
  P0:
    node: ox_inj_in
    rise: {config: prop_system.ox_inj_pumpout_dp}
ox_pump_in:
  component: junction
  fluid: {config: prop_system.initial_conditions.ox_tank.fluid}
  phase: liquid
  state0:
    P: {node: ox_pump_in}
    T: {config: prop_system.initial_conditions.ox_tank.T}
  P0:
    node: ox_pump_out
    drop: {config: prop_system.pumps.oxidizer_pump.pressure_rise_pa}
```

The relation is `P0 = referenced_P0 * (1 + relative_rise) + rise - drop`;
missing rise/drop terms default to zero. Cycles and nonpositive pressures fail
validation. These are design/initialization values, not runtime pressure BCs.
Junctions require explicit `fluid`, `phase`, and `state0.P/T`. Initial pressure
can reference a node's resolved design pressure with `{node: node_id}` or be a
number; temperature can be numeric or a config reference. The P/T values are
guesses for consistent initialization, not imposed operating conditions.
Junctions solve algebraic mass and enthalpy-flow balances without storing mass
or energy. They mix streams of one fluid and phase; different fluids or phases
at one junction are unsupported. Donor replacement after dryout/reversal is
selected between solves. With no inlet flow, temperature is anchored to the
explicit initial value (or the new donor temperature following replacement);
a fully disconnected junction also anchors pressure. No fictitious flow is added.
The energy row has a default scale of 1e6 W. Override it in the flight config
under `advanced.fluid_network.equation_scales`, using the literal key
`<node_id>.energy_rate`.
The same template pressures are used by vehicle tank sizing. An explicit initial
propellant `P` can still override its initial state without changing design pressure.

`CdA` omitted/null means size the restriction from its design flow and pressure
drop. A finite supplied `CdA` uses that area. For automatic pressurant capacity
sizing, the source must be a pressurant tank and the destination a wet tank with
liquid outlets. A fixed-area controller instead supplies `CdA` and
`target_pressure` directly. Configured pumps must each be referenced once.

A zero design pressure drop cannot size a finite restriction. To omit an injector
or another loss, remove that branch/junction from the selected template and
reconnect its neighbors explicitly; there is no hidden conditional rewiring.

Currently the vehicle output adapter supports one combustor with oxidizer/fuel
feeds and one nozzle. The network also rejects unsupported property-transport
cycles and multiple ideal regulators on the same storage volume. Templates can
assemble existing component models; new physics requires a component implementation.
