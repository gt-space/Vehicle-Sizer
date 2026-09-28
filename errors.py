"""Error classification shared by fluid and flight numerical models."""
import hashlib
import traceback


class TrialDomainError(ValueError):
    """A numerical trial is invalid or singular, not proven infeasible.

    The network adapter decides whether recovery is possible. Configuration,
    assembly and mode-coordination errors must not use this exception.
    """


class ResidualAcceptanceError(RuntimeError):
    """An integrated state failed the network's independent closure check."""


class LookupBoundsError(ValueError):
    """A property query exceeds model coverage, not a physical design limit."""
    def __init__(self, message, *, table=None, axis=None, value=None, lower=None, upper=None):
        self.details = dict(table=table, axis=axis, value=value, lower=lower, upper=upper)
        super().__init__(message)


class ModelDomainExceeded(RuntimeError):
    """A solve exhausted recovery within the supported property domain."""
    def __init__(self, message, *, time=None):
        self.details = dict(time=time)
        super().__init__(message)


class SolverConvergenceError(RuntimeError):
    """IDA could not finish a solve; not an independently rejected solution."""
    def __init__(self, operation, status, time, *, retryable=True):
        self.retryable = retryable
        self.details = dict(operation=operation, status=int(status), time=time)
        super().__init__(f'{operation} failed with SUNDIALS status {status} at t={time}')


class SolverSetupError(RuntimeError):
    """Invalid solver setup or unrecoverable native interface failure."""


class InfrastructureError(RuntimeError):
    """Required result publication or other infrastructure failed."""


class SearchFailureLimit(RuntimeError):
    """Repeated failures make continued optimization uninformative."""


def failure_details(error, *, phase=None):
    """Classify chained causes without parsing their messages."""
    chain, current = [], error
    while current is not None and all(current is not item for item in chain):
        chain.append(current)
        current = current.__cause__
    kind, fatal = 'unexpected_error', False
    selected = chain[-1]
    for classes, label, stops in (
        ((SearchFailureLimit,), 'search_failure_limit', True),
        ((KeyboardInterrupt,), 'cancelled', True),
        ((InfrastructureError, OSError), 'infrastructure_error', True),
        ((SolverSetupError, KeyError, TypeError, AttributeError, AssertionError,
          ImportError, MemoryError), 'configuration_error', True),
        ((LookupBoundsError, ModelDomainExceeded), 'table_domain_exceeded', False),
        ((ResidualAcceptanceError,), 'residual_acceptance_failure', False),
        ((SolverConvergenceError,), 'solver_nonconvergence', False),
        ((TrialDomainError,), 'trial_domain_failure', False),
        ((ValueError,), 'configuration_error', True),
    ):
        match = next((item for item in chain if isinstance(item, classes)), None)
        if match is not None:
            kind, fatal, selected = label, stops, match
            break
    frames = traceback.extract_tb(selected.__traceback__)
    location = (frames[-1].filename, frames[-1].name, frames[-1].lineno) if frames else None
    fingerprint = hashlib.sha256(repr((kind, type(selected).__name__, location)).encode()).hexdigest()[:16]
    details = {}
    for item in reversed(chain):
        details.update(getattr(item, 'details', {}))
    return dict(kind=kind, fatal=fatal, phase=phase, exception=type(selected).__name__,
                message=str(selected), fingerprint=fingerprint, details=details,
                attempts=next((item.attempts for item in chain if hasattr(item, 'attempts')), []))
