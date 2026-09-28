"""Nonfatal simulation cautions, collected separately from solver errors."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging


_active = ContextVar("simulation_cautions", default=None)
_logger = logging.getLogger(__name__)


def caution(code: str, message: str) -> None:
    """Record a caution once per simulation, counting repeated occurrences."""
    records = _active.get()
    if records is None:
        _logger.warning("%s: %s", code, message)
        return
    for record in records:
        if record["code"] == code:
            record["count"] += 1
            return
    records.append(dict(code=code, message=message, count=1))


@contextmanager
def collect_warnings():
    """Publish summaries on success or failure without warning-as-error filters."""
    records = []
    token = _active.set(records)
    try:
        yield records
    finally:
        _active.reset(token)
        for record in records:
            _logger.warning("Simulation caution [%s] (%d evaluations): %s",
                            record["code"], record["count"], record["message"])
