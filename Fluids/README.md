# Fluids

Sizes propulsion components and solves connected fluid mass, energy, and flow equations.

Inputs: tank geometry, initial states, property sources, a network template, ambient pressure, heat rates, and axial acceleration.

Network node and branch definitions select their implementation with `component`.
Template binding validates this key; sizing and assembly use it directly.

Outputs: node pressures, temperatures and inventories; branch mass flows; engine thrust; events; and operating-limit margins.

Detected thermodynamic phase transitions in volume models (condensation, evaporation, or reaching the saturated-liquid limit) raise `UnsupportedPhaseChangeError` before changing the model state. The error records the volume, fluid, transition, and event time. Optimization marks the candidate unresolved/outside model coverage and continues without adding a constraint penalty or counting it toward numerical-failure limits. This applies to all volume IDs and fluids. Propellant dryout remains an inventory-depletion event and follows its existing shutdown/flow behavior.

- `PropSystem.py`: binds config inputs, sizes components, and returns propulsion outputs.
- `FluidNetwork.py`: assembles equations and advances them with SUNDIALS.
- `FluidNode.py` / `FluidBranch.py`: storage, boundaries, flow laws, and component transitions.
- `events.py`: event definitions used for root detection and mode changes.
- `design.py`: initial states and pump sizing.


```bash
.venv/bin/python examples/run_propulsion.py Configs/Vespula.yaml --duration 1 --dt 0.1
```

Gas-generator consumption approximation
--------------------------------------

Select `Configs/templates/gas_generator_blowdown.yaml`, or add two `mass_flow`
branches from tank liquid ports to a boundary in an existing template. Configure
GG pumps with `drive: gas_generator`; their `efficiency` is pump-only. Electric
pumps retain their combined pump/motor/drive efficiency and `max_power_kw` limit.
For example, merge these fields into an existing pump-fed propulsion config:

```yaml
prop_system:
  template: Configs/templates/gas_generator_blowdown.yaml
  pumps:
    fuel_pump:
      drive: gas_generator
      pressure_rise_pa: 1000000
      efficiency: 0.7
      gas_CdA: 0.0001
    oxidizer_pump:
      drive: gas_generator
      pressure_rise_pa: 1000000
      efficiency: 0.7
      gas_CdA: 0.0001
  gas_generators:
    main:
      pumps: [fuel_pump, oxidizer_pump]
      fuel_branch: GG_FUEL
      oxidizer_branch: GG_OX
      efficiency: 0.6
      turbine_outlet_pressure_pa: 100000
      mixture_ratio: 0.5
      stiffness: 0.2
```

Sizing runs in ordered passes: ordinary design flows, pumps, GG demands,
prescribed-flow validation, restrictions, then pressurant feeds. The GG pressure
is the lower upstream design pressure divided by `1 + stiffness`. Combustion
properties are evaluated at that pressure and GG mixture ratio. Temporarily,
`cp = gamma * R / (gamma - 1)`; replacing that placeholder with tabulated chamber
cp will not change the turbine-sizing equation. Turbine demand is shaft power
from its pumps divided by `efficiency * cp * isentropic_delta_T`.

Drains remove donor enthalpy as well as mass through existing node balances.
Their setpoints remain fixed during flight. Liquid-only drains stop independently
at donor dryout; with post-shutdown solving enabled, the other tank continues to
drain at its prescribed rate. Normal stop-at-shutdown behavior freezes the whole
network. `PropSystem.gg_sizing` and `SimResult.gg_sizing` expose design calculations.
Direct tank drains account for consumption only; downstream GG feeds requiring
additional pump throughput are rejected until coupled design-flow sizing exists.
The model includes no turbine dynamics, GG exhaust thrust, or GG hardware mass.

Network state and rollback
--------------------------

`initialize()` and `update()` use one bounded retry/restore loop. Initialization
sets the starting time; updates can also initialize an unused network. Each
operation has one retry budget, including initialization and any subsequent
advance. `commit=False` advances an isolated runtime copy and closes its solver,
leaving the accepted network and native solver history intact.

Checkpoints copy the fields declared in `FluidNetwork._runtime_fields`, including
component runtime attributes, numerical layouts, constraint extrema, pressure
tracking, event counts, and effective solver controls. Restore is reusable.
Component definitions, geometry, property providers, and monitor callbacks are
shared; fixed definitions and solver options must remain unchanged during a run.
`effective_horizon` holds retry adjustments independently of configured options.
New component attributes participate automatically in rollback. New network-level
runtime fields belong in `_runtime_fields`; subclasses can extend that tuple.

`NetworkState` lives with the other records in `FluidState.py` and owns snapshot
serialization. Returned `node`/`branch` data are detached from solver state and
are the canonical access points; redundant `td_state` and `state` exports have
been removed. Constraint monitors receive a detached `NetworkState` with `nodes`
and `branches`.

`events` and `numerical_remainders` contain only records from the current operation;
`event_counts` remains cumulative. Event numbering is assigned once by the network
and rolls back with the operation. Numerical remainders are derived from those
events rather than stored separately. Consumers retaining complete history append
the returned events; the flight history already does this. Saved output snapshots
remain valid after later advances, previews, and restores.

Every update also returns `samples`: ordered `(time, snapshot)` pairs at the
start, both sides of events, and the endpoint. Repeated times preserve a jump;
coincident events settle before the final right-hand snapshot. Failed attempts
discard their sample list. `PropSystem` converts these to detached `FluidOut`
leaves with empty `samples`, using the same conversion as its final output and
shutdown `before` record. These samples drive flight interpolation; they are
not optional and do not contain every internal IDAS step.

Assessments use immutable `constraint_records` throughout network and propulsion
outputs; monitors return the same record type. Margin and measurement time roll
back together. Initial assessment records are seeded once, instead of being
merged into every propulsion output. Pump-domain validation remains active even
when optional assessment is disabled. External-heating nodes and pump curves
are identified once during propulsion construction.

Sizing and runtime gas restrictions use the same `compressible_mass_flux()`
equation. Invalid physical inputs raise `ValueError` for sizing; the runtime
adapter classifies them as recoverable `TrialDomainError` solver inputs.

Pressurant-feed sizing uses `eol_temperature` (kelvin) on each regulator under
`prop_system.regulator.<branch_id>`. The representative sizing temperature is the
mean of initial pressurant temperature and this EOL value. Automatically sized
bang-bang feeds use the same field in their branch configuration. Explicitly
supplied CdA values do not require it. Tank acceptance limits do not affect sizing.

The regulator flight configs keep sizing inputs only. Their corresponding
optimizer configs supply mission/design constraints and tank `min_temperature` /
`min_pressure` limits through `evaluation_overrides`. Standalone flights retain
physical model validity checks without applying optimizer acceptance limits.

Pressure-fed feed sizing
------------------------

For each direct tank-to-injector-inlet restriction, specify exactly one non-null
input in the vehicle config (pressure drop in Pa or effective CdA in m²):

```yaml
prop_system:
  ox_tank_inj_dp: 275790.291727
  # ox_system_cda: 0.00032706944491738433
  fuel_tank_inj_dp: 275790.291727
  # fuel_system_cda: 0.0001776871392414737
```

Comment out the corresponding ΔP line to use CdA. The injector restriction is
separate. ΔP inputs retain automatic restriction sizing. CdA inputs derive the
required tank pressure from design flow and pressure-dependent liquid density
before sizing tank walls, inventories, and regulator feeds. They also set the
runtime feed CdA directly. Both modes retain explicit initial tank pressure
overrides; those affect initial density and wall-sizing pressure as before.

CdA mode uses the template's `OX_TANK_INJ` / `FUEL_TANK_INJ` liquid loss branches
and their tank-pressure rise relationships. Do not also specify CdA on those
template branches. Unsupported wiring or out-of-domain pressure requirements
raise sizing errors. Custom callers of `prepare_design_inputs` must pass
`fluid_properties` and `combustion_properties` when using CdA inputs; main,
optimizer, and the propulsion example already do so. Source configs are not mutated.

Tank wall allowance and optimizer minimum
-----------------------------------------

`advanced.tank_wall_thickness_allowance` adds a thickness in metres to automatically
pressure-sized propellant and metal pressurant walls. It defaults to zero and must
be finite and nonnegative. Explicit wall overrides are final nominal thicknesses;
empirical COPV walls are unchanged. The final wall controls tank geometry and mass.

Optimizer `evaluation_overrides` can set `tanks.<id>.min_wall_thickness` and declare
`tank.<id>.wall_min` in `hard_constraints`. This checks the pressure-calculated
`required_wall_thickness` **before** the allowance, without clamping it. The current
pressure-fed settings use a 0.000254 m (10 thou) allowance and 0.00254 m (100 thou)
minimum calculated walls for LOX and fuel. These minimums are optimizer-only.
