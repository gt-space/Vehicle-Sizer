# Fluids

Sizes propulsion components and solves connected fluid mass, energy, and flow equations.

Inputs: tank geometry, initial states, property sources, a network template, ambient pressure, heat rates, and axial acceleration.

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
