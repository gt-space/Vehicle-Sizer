"""Error classification shared by fluid and flight numerical models."""
import hashlib
import traceback


class GeometryError(ValueError):
    """Geometry cannot be represented by the model or search coordinates."""
    def __init__(self, margins):
        self.constraints = margins  # Signed bounds used by optimizer decoding.
        self.details = dict(margins=margins)
        super().__init__(f"Unsupported geometry: {margins}")


class SimulationIncompleteError(RuntimeError):
    """Integration stopped before reaching apogee."""
    def __init__(self, result):
        self.details = dict(termination=result.termination, time_s=result.final_time)
        super().__init__(f"Flight did not reach apogee: {result.termination} at t={result.final_time:g} s")


class EvaluationFailure(RuntimeError):
    """One simulation failure contract, including inputs, partial state and cautions."""
    def __init__(self, phase, config, partial_result=None, *, warnings=()):
        self.phase = phase
        self.config = config
        self.partial_result = partial_result
        self.warnings = list(warnings)
        super().__init__(f"Simulation failed during {phase}")

    @property
    def diagnostic(self):
        return dict(failure_details(self, phase=self.phase), warnings=self.warnings)

    def __str__(self):
        reason = f": {type(self.__cause__).__name__}: {self.__cause__}" if self.__cause__ else ''
        return super().__str__() + reason


def format_failure(error):
    """The same readable failure reason for single runs and sweep cases."""
    detail = failure_details(error, phase=getattr(error, 'phase', None))
    message = (f"Simulation failed [{detail['kind']} / {detail['exception']}]"
               f" during {detail['phase'] or 'evaluation'}: {detail['message']}")
    for warning in getattr(error, 'warnings', ()):
        message += f"\nWarning [{warning['code']}]: {warning['message']}"
    return message


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


class UnsupportedPhaseChangeError(ModelDomainExceeded):
    """A located volume transition requires physics outside model coverage."""
    def __init__(self, node_id, fluid, from_phase, to_phase, event, time):
        super().__init__(f"Volume {node_id!r} ({fluid}) requires unsupported phase change "
                         f"from {from_phase} to {to_phase} at t={time:g} s ({event})",
                         time=float(time))
        self.details.update(node_id=node_id, fluid=fluid, from_phase=from_phase,
                            to_phase=to_phase, event=event, time_s=float(time))


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
        ((UnsupportedPhaseChangeError,), 'unsupported_phase_change', False),
        ((GeometryError,), 'geometry_error', False),
        ((SimulationIncompleteError,), 'incomplete_flight', False),
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
