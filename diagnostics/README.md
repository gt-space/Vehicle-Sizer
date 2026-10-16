# Diagnostics

`errors.py` defines model, solver, infrastructure, and search failures, the common
`EvaluationFailure` wrapper, classification, and human-readable error reporting.
`constraints.py` contains optimizer assessments: configuration limits, geometry
and tank margins, and constraint records. Negative margins do not throw errors
or stop integration. `warnings.py` collects nonfatal cautions independently.

Import the specific module, for example:

```python
from diagnostics.errors import UnsupportedPhaseChangeError
from diagnostics.errors import EvaluationFailure
from diagnostics.constraints import configured_limits
from diagnostics.warnings import caution
```

`simulate(cfg)` builds and runs a vehicle once, returning a completed flight or
raising `EvaluationFailure`. Its `diagnostic` includes the underlying exception
type, failure kind, phase, details and warning records; `partial_result` is for
debugging, not a completed output. A time limit or no-ascent termination is an
`incomplete_flight` error. The optimizer calls `simulate(cfg,
assess_constraints=True)` to attach assessments using that same physical path.

The AoA model boundary (up to 15 degrees, subject to the aero table) is independent
of `constraints.max_aoa_deg`. A stricter configured limit affects optimizer
feasibility, never integration. The existing low-dynamic-pressure ballistic
mode skips aero tables and therefore their angle bounds. Numerical trial-domain
violations can retry smaller steps; exhaustion is a reported failure.

Packaging, tank temperature/pressure floors, pump power margins and mission
targets are assessed only for optimization. Invalid geometry that cannot be
represented on the structural grid, unsupported phase transitions, invalid
model inputs and lookup-table bounds remain errors for every caller.

`constraint_records` is the sole assessment mapping in `FluidNetwork`, `FluidOut`,
and `SimResult`. Each immutable `ConstraintRecord` carries margin, source, phase,
units, required status, and the time of that measurement. `merge_records()` keeps
the worst assessed value and its time together; equal margins retain the first
measurement. Returned mappings are detached and records can safely be shared.
Network checkpoints restore assessments through the ordinary runtime contract.

Assessment preparation seeds geometry, pump, and initial tank records once.
Monitors return records with metadata; the network timestamps accepted samples.
Mission finalization adds its records directly. The former `constraints`,
`geometry_constraints`, and `constraint_times` result fields are removed. Read
`result.constraint_records[name].margin` or `.time`; serialized records expose
those same fields. `feasible` considers required records only: a negative margin
is false, missing assessment is unknown, and optional checks do not veto a design.
`accepted` additionally requires completed ascent, completed burn, and mission
assessments. Execution failures remain separate from assessment outcomes.
