"""Keep native propulsion resources alive until orderly teardown after each test."""
import pytest


@pytest.fixture(autouse=True)
def close_propulsion_systems(monkeypatch):
    from Fluids.PropSystem import PropSystem

    initialize = PropSystem.__init__
    systems = []

    def tracked(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        systems.append(self)

    monkeypatch.setattr(PropSystem, '__init__', tracked)
    yield
    for system in reversed(systems):
        system.close()
