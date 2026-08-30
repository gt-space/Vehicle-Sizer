from __future__ import annotations
import inspect
from typing import TYPE_CHECKING
from .State import State
from .Sizer import Sizer
if TYPE_CHECKING:
    from .Network import Network



class Component:

    def __init__(self, name: str, network: Network, sizer: Sizer = None):
        self.setup()


    def setup(self) -> None:
        arguments = inspect.currentframe().f_back.f_locals.copy()
        arguments.pop("self")

        name = arguments.pop("name")
        network = arguments.pop("network")
        self.initialize_component(name, network)
        self.sizer = arguments.pop("sizer", None)

        for name, value in arguments.items():
            setattr(self, name, self.initialize_attribute(value))

    def initialize_component(self, name: str, network: Network) -> None:
        self.name = name
        self.network = network
        self.network.add_component(self)


    def initialize_attribute(self, value=None):
        if isinstance(value, Sizer): # Sizer objects should not be turned into States
            return value
        if isinstance(value, State):
            return value
        return State(value)


    def size(self) -> None:
        if self.sizer is not None:
            self.sizer.size(self)
            

    def evaluate(self) -> None:
        pass

    @property
    def dynamics(self) -> list[tuple]:
        return []

    @property
    def balances(self) -> list[tuple]:
        return []

    def __str__(self) -> str:
        lines = [f"{self.__class__.__name__}: {self.name}"]
        for name, value in vars(self).items():
            if name in ("name", "network"):
                continue
            if isinstance(value, State):
                value = value.value
            elif isinstance(value, Sizer):
                value = value.__class__.__name__
            lines.append(f"  {name}: {value}")
        return "\n".join(lines)