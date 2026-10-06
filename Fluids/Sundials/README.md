# SUNDIALS solver

Connects the fluid network to the `sundials4py` IDAS solver.

Inputs: residual functions, state and derivative vectors, tolerances, root functions, and Jacobian structure. Outputs: accepted states, event roots, solver statistics, or typed failures.

`ida_session.py` owns solver sessions and consistent initialization. `jacobian.py` builds numerical Jacobians using dependency groups.

Used automatically by `FluidNetwork`. Run checks from the project root:

```bash
.venv/bin/python -m pytest tests/test_fluid_jacobian.py tests/test_fluid_network.py -q
```
