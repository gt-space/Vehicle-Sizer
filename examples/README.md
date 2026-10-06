# Examples

Run these commands from the project root.

Propulsion at fixed ambient pressure, with JSON output:

```bash
.venv/bin/python examples/run_propulsion.py Configs/Vespula.yaml --duration 1 --dt 0.1
```

The input must use table properties and tank heating set to `None` or `DensityPowerLaw`.

Parallel optimizer search with per-candidate records, timeouts, and saved progress:

```bash
.venv/bin/python examples/run_optimizer_search.py --config Configs/optimizer_pressure_fed.yaml --output outputs/parallel_search_01 --workers 4
```

Repeat the command with the same output directory to resume if source files and configs are unchanged. Outputs include candidate configs, results, failure reports, progress, and sampled timing records.

`search_timing.py` supplies timers used by the search runner; it has no standalone command.
