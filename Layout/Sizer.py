import inspect
from abc import ABC, abstractmethod

class Sizer(ABC):

    def setup(self) -> None:
        arguments = inspect.currentframe().f_back.f_locals.copy()
        arguments.pop("self")
        for name, value in arguments.items():
            setattr(self, name, value)


    @abstractmethod
    def size(self, component):
        pass