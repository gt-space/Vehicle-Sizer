from __future__ import annotations
import inspect
from typing import TYPE_CHECKING
from .State import State
if TYPE_CHECKING:
    from .Network import Network



class Component:

    def __init__(self, name: str, network: Network):
        self.setup()


    def setup(self) -> None:
        arguments = inspect.currentframe().f_back.f_locals.copy()
        arguments.pop("self")

        name = arguments.pop("name")
        network = arguments.pop("network")
        self.initialize_component(name, network)

        for name, value in arguments.items():
            setattr(self, name, self.initialize_attribute(value))

    def initialize_component(self, name: str, network: Network) -> None:
        self.name = name
        self.network = network
        self.network.add_component(self)


    def initialize_attribute(self, value=None):
        if isinstance(value, State):
            return value
        return State(value)


    def evaluate(self) -> None:
        pass

    @property
    def dynamics(self) -> list[tuple]:
        return []

    @property
    def balances(self) -> list[tuple]:
        return []
