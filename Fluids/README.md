# Fluids

Sizes propulsion components and solves connected fluid mass, energy, and flow equations.

Inputs: tank geometry, initial states, property sources, a network template, ambient pressure, heat rates, and axial acceleration.

Outputs: node pressures, temperatures and inventories; branch mass flows; engine thrust; events; and operating-limit margins.

- `PropSystem.py`: binds config inputs, sizes components, and returns propulsion outputs.
- `FluidNetwork.py`: assembles equations and advances them with SUNDIALS.
- `FluidNode.py` / `FluidBranch.py`: storage, boundaries, flow laws, and component transitions.
- `events.py`: event definitions used for root detection and mode changes.
- `design.py`: initial states and pump sizing.


```bash
.venv/bin/python examples/run_propulsion.py Configs/Vespula.yaml --duration 1 --dt 0.1
```
