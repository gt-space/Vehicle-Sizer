# Fluid helpers

- `templates.py`: resolves config references, design pressures, and network wiring.
- `pump_curve.py`: loads pump curves and evaluates scaled head and efficiency.
- `pressure_tracking.py`: accumulates tank-pressure error during flight.

Inputs are config/template data, pump operating points, and accepted fluid states. Outputs are resolved network definitions, pump performance, and pressure-error totals.

These helpers run through `PropSystem` and the optimizer.