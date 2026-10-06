"""Component event contract: positive margins are valid; falling zero is a root.

Components return only armed events from events(context), and apply_event maps
their local values/mode at a root. Context supplies time, raw values/states,
rates, error scales, options and a lazy evaluated() network snapshot. Guards
must be side-effect free. Negate a rising physical threshold to use this common
sign convention. No event names or component types belong in the dispatcher.

Keep guard names stable during an integration segment. For state-dependent
arming, expose an activation guard, then arm the next guards in apply_event.
Use continuous margins for root localization; a boolean jump cannot identify
an exact crossing time. at_zero controls initial/equal-boundary activation;
tolerance permits settling roundoff; coincident includes a nearby event when
another root is found. terminal supplies a network stop reason.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    value: float
    at_zero: bool = False
    tolerance: float = 0.0
    terminal: str | None = None
    coincident: bool = False
