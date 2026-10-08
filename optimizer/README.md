# Optimizer

`core.py` defines candidate decoding, scoring, evaluation, verification, and the
`optimize()` API. `workers.py` supervises process workers, candidate timeouts,
recycling, and resumption. `timing.py` supplies optional nested CPU/wall profiling.

Run from the project root:

```bash
.venv/bin/python -m optimizer Configs/optimizer_epump.yaml --output outputs/epump_search_01 --workers 4 --max-evaluations 100
```

Use `--profile` to save candidate timings, or `--resume` to replay saved results
and continue with the original settings, budget, and source files. The default
is serial evaluation; worker controls can also be specified in the search config.

Python callers use `from optimizer.core import optimize`. The worker entry point
is `python -m optimizer.workers`; the coordinator launches it automatically.
Source fingerprints include the packages used by evaluations and exclude `examples`.

Candidate records in `evaluations.jsonl`, worker `result.json`, and `best.json`
include `gg_sizing` keyed by GG name. Each entry reports required shaft power [W],
turbine inlet/outlet and upstream feed pressures [Pa absolute], oxidizer/fuel
mixture ratio, and total/fuel/oxidizer mass flows [kg/s]. The record is empty when
there is no GG or GG sizing was unavailable. Available sizing is retained for
later preflight rejections and partial flight results.
