# Simulation and optimizer failure policy

Shared exception definitions are in root `errors.py`; import them directly.
Physical constraint exceptions remain in `constraints.py`. Nonfatal cautions
are collected by `warning.py` and do not change candidate scores.

## Numerical recovery

`LookupBoundsError` includes table, axis, value and limits. In an IDA residual or
numerical-Jacobian callback it is recoverable, allowing IDA to retreat from an
invalid trial. Root callback errors are not recoverable. If IDA cannot recover,
the failure is reported as `ModelDomainExceeded` / `table_domain_exceeded`.
Direct lookup failures during sizing are also unsupported-domain outcomes. No
other property coordinates are silently clamped or extrapolated.

`SolverConvergenceError` preserves the IDA operation, status and last accepted
session time. Convergence/error-test/constraint/line-search failures may retry
from the network checkpoint with a smaller maximum step; initialization may retry
with a shorter initialization horizon. Accuracy/work-limit failures are reported
without repeating the same demand or tightening tolerances. Setup/illegal-input
and unrecoverable native-interface statuses raise fatal `SolverSetupError`.
Status categories follow the
[SUNDIALS IDAS constants](https://sundials.readthedocs.io/en/develop/idas/Constants_link.html).

`ResidualAcceptanceError` means IDA returned but the independent residual check
rejected the state. Its retries tighten integration tolerances and reduce maximum
step size. Both policies use the network's `residual_retries` budget, interpreted
as additional attempts. Failed updates restore the checkpoint, including physical
state, events and constraints. Final errors retain attempt diagnostics, while
successful recoveries retain `retry_diagnostics`. Initialization and interval
updates each have their own bounded retry loop; the worker wall timeout also
bounds an entire candidate.

These errors are not evidence of physical infeasibility. `OperatingInfeasible`
continues to represent a converged physical-limit violation. Incomplete flights
retain missing mission metrics and cannot become feasible candidates.

## Candidate results

`EvaluationFailure` retains configuration, phase, partial result and chained cause.
The optimizer records structured `failure_details` alongside full traceback text.
Failure kinds distinguish table coverage, solver nonconvergence, residual closure,
trial-domain failure and unexpected runtime error. Each unresolved candidate gets
the existing score 3; completed and feasible categories keep their existing scores.
Unknown runtime exceptions may end one candidate, but repeated errors are bounded.
Invalid inputs/invariants, missing dependencies, solver setup and required I/O
failures stop the run instead of being scored as nonconvergence.

## Process isolation and publication

Use `examples/run_optimizer_search.py` for crash-isolated optimizing runs.
The in-process `optimizer.optimize()` handles Python errors but cannot contain
native memory corruption. Python exception handling does not repair that bug.

The supervisor recognizes signed and unsigned Windows NTSTATUS crash codes as
well as negative POSIX signal exits. `failure_policy.native_retries` accepts 0 or
1, defaulting to 0; the regulated search configuration explicitly selects 1.
A retry always uses a fresh process. Attempts and crash logs are retained even
when recovery succeeds. Exhausted crashes/timeouts become unresolved candidates.
Fatal worker diagnostics stop the coordinator. Results are published only after
native cleanup, preventing a destructor crash from being assigned to a later job.

JSON publication uses unique temporary files and atomic replacement. Replacement
PermissionError is retried for up to three seconds. Optional status/progress output
can fail with a caution; required result writes raise InfrastructureError. Physics
is not rerun merely because publication is blocked.

## Circuit breaker and reproducibility

`failure_policy` settings (defaults):

```yaml
failure_policy:
  consecutive_limit: 5
  window: 50
  rate_limit: 0.2
  repeated_limit: 3
  native_retries: 0
```

The monitor stops after five consecutive numerical/worker failures, a rate above
20% in a full 50-candidate window, or three matching unexpected/native error
fingerprints. Recovered native crashes count too. Physical and table-coverage
rejects are tracked separately and do not count as unexplained numerical failures.
Thresholds are operational defaults, not numerical guarantees.

Parallel results are checked in candidate order rather than completion order.
The circuit breaker is checked after a submitted generation batch completes.
Fatal errors or cancellation stop active workers without inventing scores for
cancelled candidates. `stopped.json` records why the run ended. Candidate files
remain authoritative; the existing seeded DE replay reconstructs progress using
unchanged saved results/settings/source fingerprints. There is no new binary
snapshot of native solver state. A persistent failure limit will stop the same
replay again until the cause/policy is addressed; changed physics requires a new run.

Flat rejection scores still cannot trigger the DE convergence criterion. Watch
failure kinds/rates, feasible count and constraint improvement, not just best
score. Neither process isolation nor retries prove that failed search regions
contain no good designs.

## Known limitation

The native access violation remains unresolved. The expanded integration tests
reproduced it in an existing dryout test. A separate diagnostic harness under
`outputs/error_handling_validation_20260928` uses the prior native-lifetime
quarantine to check Python recovery behavior; that memory-retaining workaround
is not enabled in production.
