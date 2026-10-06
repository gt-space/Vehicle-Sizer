"""Component event contract: positive margins are valid; falling zero is a root."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    value: float
    at_zero: bool = False
    tolerance: float = 0.0
    terminal: str | None = None
    coincident: bool = False
