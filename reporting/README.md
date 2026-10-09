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

Plot saved optimizer results from the project root with
`python reporting/plot_optimizer_convergence.py outputs/pressure_fed_search_02`.
This writes `convergence.png` and `convergence.svg` in the search directory;
use `--output` to choose another output prefix or `--window` to set the
constraint violation rolling window.

Plot existing sensitivity sweep results with:

```bash
python -m reporting.plot_sensitivities outputs/sensitivity/my_sweep --list-fields
python -m reporting.plot_sensitivities outputs/sensitivity/my_sweep \
  --x tanks.press_tank.volume --y apogee --xscale 0.001 --yscale 1000 \
  --xlabel "COPV volume (L)" --ylabel "Apogee (km)" --output /tmp/sweep.png
```

See [the sensitivity plotting documentation](../sensitivity/README.md) for
history plots, derivatives, multiple axes, and the Python API.
