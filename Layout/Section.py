from __future__ import annotations
import inspect
from typing import TYPE_CHECKING
from .State import State
if TYPE_CHECKING:
    from .Vehicle import Vehicle


class Section:

    required_states = ("mass", "length", "EI")

    def __init__(self, name: str, vehicle: Vehicle):
        # the stacking order of sections in the vehicle is the same as the instantiation order
        self.setup()


    def setup(self) -> None:
        arguments = inspect.currentframe().f_back.f_locals.copy()
        arguments.pop("self")

        name = arguments.pop("name")
        vehicle = arguments.pop("vehicle")
        self.initialize_section(name, vehicle)

        for name, value in arguments.items():
            setattr(self, name, value)

        for name in self.required_states:
            setattr(self, name, State())


    def initialize_section(self, name: str, vehicle: Vehicle) -> None:
        self.name = name
        self.vehicle = vehicle
        self.vehicle.add_section(self)


    def evaluate(self) -> None:
        pass


    def check_properties(self) -> None:
        missing = [name for name in self.required_states if not getattr(self, name).is_assigned]
        if missing: raise ValueError(f"Section '{self.name}' did not assign: {', '.join(missing)}")
