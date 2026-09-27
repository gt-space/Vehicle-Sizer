# Fluid and propulsion model

This is the sole fluid package used by `simulation.py`, `optimizer.py` and the
examples. Integration uses SUNDIALS IDAS through `sundials4py` (validated on
Python 3.12 with version 7.9.0). No alternate solver selection is required.

## Responsibilities

- `PropSystem.py`: resolves the selected template, sizes engine/restrictions/pumps,
  derives initial inventories from vehicle geometry and P/T, assembles the network,
  and returns the flight-facing `FluidOut`. Owns sizing and operating constraints.
- `templates.py`: loads one template, binds config references and resolves design
  pressure relations. [Template schema](../Configs/templates/README.md).
- `FluidNetwork.py`: indexes unknowns/equations, routes transported properties,
  evaluates the assembled DAE, initializes consistent states, integrates,
  localizes events, applies mode changes and retains accepted snapshots.
- `FluidNode.py`: storage mass/energy balances, EOS and volume closures, junction
  continuity, combustion properties, outlet states and node event functions.
- `FluidBranch.py`: conservation-of-momentum models, transported streams and valve,
  pump and nozzle modes. Trial evaluation never commits a discrete transition.
- `FluidState.py`: `FluidState`, `NodeState` and `BranchState` data records. Trial
  unknowns and derived properties remain distinct; records do not solve physics.
- `ida_session.py`: native IDAS callbacks, solver lifetime and restarts.
- `jacobian.py`: equation dependency structure and colored finite differences.
- `errors.py`: recoverable trial-domain and accepted-residual error types.
- `design.py`, `pump_curve.py`, `FluidsDef.py`: shared sizing and physics helpers.

`FluidTables/PropertyModels.py` supplies CoolProp/CEA or table-backed properties.
The exact-coordinate property cache shares state/derivative lookup work; complete
storage-node evaluations are not cached. Wet-tank thermodynamic response uses
an algebraic solve with explicit validity checks and no numerical fallback.

## Configuration and use

Use one of the four `Configs/flight_*.yaml` files for pump-fed/pressure-fed,
regulator/bang-bang systems. Six wiring templates also include true blowdown.
The config selects wiring only through `prop_system.template`. Initial P/T and
fluid identities belong in `prop_system.initial_conditions`; missing propellant
pressure defaults to its template design pressure. COPV pressure comes from its
single tank `design_pressure` input. Inventories are derived on each build.

```python
from simulation import simulate
from Configs.loader import load_config
result = simulate(load_config('Configs/flight_pump_fed_regulator.yaml'))
```

For direct propulsion work, use `PropSystem` as a context manager or call
`close()` when done. `update(None, atmosphere, heat)` initializes; a positive dt
advances the network. `commit=False` previews and restores the accepted snapshot.
The simulation API closes the native solver after each candidate automatically.

## Integration settings

Set options under `advanced.fluid_network`. Main controls include:

- `rtol`: relative state error tolerance. Template defaults are `1e-4` for
  regulators, `1e-5` for bang-bang and `1e-7` for blowdown.
- `atol`: mapping of qualified variable names to absolute state error tolerances;
  unspecified variables use the per-variable defaults in `FluidNetwork.py`.
- `max_step`: maximum internal integration step, separate from `simulation.dt`.
- `residual_tolerance`: maximum accepted scaled equation residual.
- `jacobian`: `colored` (default) or `dense`; dependency structure
  must match the active component equations and modes.
- `residual_retries`, `retry_rtol_factor`, `retry_rtol_floor`: bounded retry policy
  after a residual acceptance failure. Each retry restores the same snapshot.
- `event_time_tolerance`, `max_events`: event localization/grouping tolerance and
  transition limit, preventing unlimited event cycling.

State integration error and equation residual tolerances check different things.
Residual scales normalize physical equations; tolerances do not change physics.
For the complete supported settings and validation, see `FluidNetwork.__init__`.

## Events and coupling

IDA localizes component root functions. The network applies transitions at the
root, updates mode-dependent layouts/property routing, computes consistent
initial conditions and restarts integration. Donor direction remains fixed
inside each solve. Invalid trial properties are recoverable solver failures;
invalid accepted states remain errors with diagnostics.

Tank dryout, chamber shutdown, flow reversal, regulator limits, bang-bang
switching and saturation transitions are handled explicitly. The network can
continue passive flow after shutdown, or freeze flows and inventories when
`simulation.fluid_solve_post_shutdown: false`. Flight then coasts to apogee.
`simulation.fluid_stop_at_triple_point` optionally freezes at the supported
saturation-domain boundary. Constraint minima use accepted fluid states.

Flight snapshots each coupled interval. Rail exit and shutdown split the flight
interval and repeat from the snapshot; thermal state commits only on acceptance.
Native IDA allocations are reused only when compatible. Required restarts and
consistent initialization remain in place when equations or boundaries change.

Run `python -m pytest -q` for component, conservation, event, rollback, template,
property, geometry, optimizer and flight coverage. Use
`python examples/run_propulsion.py --help` for a fixed-ambient propulsion run.
