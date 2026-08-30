from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Network import Network
    from .Section import Section

class Vehicle:

    def __init__(self, name:str):
        self.name = name
        self.network = None
        self.sections: list[Section] = []

    def add_network(self, network:Network): # extende to multiple networks in a single vehicle
        if self.network is None:
            self.network = network

    def add_section(self, section:Section): # change to have the order of the sections matter
        if section not in self.sections:
            self.sections.append(section)
