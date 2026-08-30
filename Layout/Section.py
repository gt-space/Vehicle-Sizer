from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Vehicle import Vehicle


class Section(ABC):

    def __init__(self, name: str, vehicle: Vehicle):
        # the stacking order of sections in the vehicle is the same as the instantiation order
        self.setup()


    def setup(self) -> None:
        arguments = inspect.currentframe().f_back.f_locals.copy()
        arguments.pop("self")

        name = arguments.pop("name")
        vehicle = arguments.pop("vehicle")
        self.initialize_section(name, vehicle)


    def initialize_section(self, name: str, vehicle: Vehicle) -> None:
        self.name = name
        self.vehicle = vehicle
        self.vehicle.add_section(self)

    def evaluate(self) -> None:
        pass

    @property
    @abstractmethod
    def length(self): pass

    @property
    @abstractmethod
    def mass(self): pass

    @property
    @abstractmethod
    def EI(self): pass