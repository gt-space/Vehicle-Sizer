"""Error classification shared by the SUNDIALS component models."""


class TrialDomainError(ValueError):
    """A numerical trial is invalid or singular, not proven infeasible.

    The network adapter decides whether recovery is possible. Configuration,
    assembly and mode-coordination errors must not use this exception.
    """


class ResidualAcceptanceError(RuntimeError):
    """An integrated state failed the network's independent closure check."""
