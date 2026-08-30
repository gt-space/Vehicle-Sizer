from __future__ import annotations
from typing import TYPE_CHECKING
from Solver import solve
if TYPE_CHECKING:
    from .Network import Network
    from .Section import Section

class Vehicle:

    def __init__(self, name:str):
        self.name = name
        self.network = None
        self.sections: list[Section] = [] # the stacking order of sections in the vehicle is the same as the instantiation order

    def add_network(self, network:Network): # extende to multiple networks in a single vehicle
        if self.network is None:
            self.network = network

    def add_section(self, section:Section): # change to have the order of the sections matter
        if section not in self.sections:
            self.sections.append(section)

    def size(self) -> None:
        #for network in self.networks: # eventually, when there are multiple networks
        for component in self.network.components:
            component.size()

    def fly(self, dt: float, t_final: float) -> dict:
        return solve(self, dt, t_final)
