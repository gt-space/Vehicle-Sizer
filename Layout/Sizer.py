from abc import ABC, abstractmethod


class Sizer(ABC):

    @abstractmethod
    def size(self, component):
        pass