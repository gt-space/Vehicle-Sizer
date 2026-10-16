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

`Evaluator.evaluate(candidate, progress=...)` decodes search coordinates once and
returns the resolved configuration, `SimResult`, and scored record. The serial
optimizer callable owns its budget, incumbent, and JSONL log; workers publish the
returned record directly. Each evaluation saves `candidate.yaml` before simulation
so native failures retain their inputs. Workers reuse property/aero models and
retain process deadlines, recycling, source checks, and resume support.

Verification calls the same evaluator with the saved configuration and
`verification=True`. It bypasses decoding, applies `verification_dt_factor`, enables
loads, and checks atmosphere coverage in both execution modes. A failed simulation
raises; a completed design that violates constraints is recorded as rejected.
Verification does not consume the search budget or replace the search incumbent.

Simulation owns propulsion cleanup and returns recovered solver attempts in
`SimResult.fluid_retries`, including partial flight results. Evaluation records expose
these as `recovered_fluid_retries`; load peaks and failure classification also use the
same record construction in serial and worker runs. Per-worker `evaluations.jsonl`
and `best.*` files are no longer produced; `result.json` is the persisted worker
record and the coordinator writes the search-wide log and winner.

Search preparation validates soft-penalty settings once; scoring checks changing
measurements. `candidate_score()` returns `(score, violations, soft_penalties)` so
the evaluation record reuses the exact penalties used in the objective. Constraint
scoring consumes `ConstraintRecord` directly; there is no raw-margin fallback.
