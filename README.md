# Vehicle-Sizer

## Run

Use Python 3.12 with sundials4py, NumPy, SciPy,
CoolProp, RocketCEA, h5py, PyYAML, pandas and matplotlib. 

```bash
python main.py Configs/flight_pump_fed_regulator.yaml
python examples/run_propulsion.py Configs/flight_pressure_fed_bang_bang.yaml --duration 1 --dt 1
python optimizer.py Configs/optimizer.yaml --output outputs/optimization
python -m pytest -q
```

## Simulation API

```python
from Configs.loader import load_config
from simulation import simulate

cfg = load_config("Configs/flight_pressure_fed_regulator.yaml")
result = simulate(cfg)
print(result.dry_mass, result.initial_mass, result.apogee)
```

`compute_loads=True` evaluates load distributions 
`simulation.fluid_stop_at_triple_point: true` freezes the entire
fluid network when any stored fluid with saturation data reaches the lower
saturation-domain boundary, never turn this off, I don't know how to get around this as it would 
require modeling solids - gross'

Example mission objective and constraints
```yaml
constraints:
  goal_apogee: 10000.0       # m
  max_burn_duration: 60.0   # s
  min_rail_twr: 1.2        
```


