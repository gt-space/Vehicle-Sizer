# Reporting

`run_report.py` collects design information and prints simulation summary tables.
`flight_plots.py` renders trajectory, loads, valve, propulsion, and thermal plots.
`main.py` calls these modules after a simulation; simulations collect design
summaries when history recording is enabled.

Python callers use:

```python
from reporting.run_report import build_run_tables
from reporting.flight_plots import plot_flight
```

Run a flight from the project root with `python main.py Configs/Vespula.yaml`.
Output locations remain controlled by the configuration's `simulation.output`
and `simulation.plot` entries.
