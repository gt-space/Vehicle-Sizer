from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass
from .State import State
from Solver import solve

from Physics import get_atmospheric_pressure

if TYPE_CHECKING:
    from .Network import Network
    from .Section import Section

@dataclass
class LaunchInputs:
    initial_altitude: float

    @property
    def initial_atmospheric_pressure(self):
        return get_atmospheric_pressure(self.initial_altitude)


class Vehicle:

    def __init__(self, name:str, launch_inputs:LaunchInputs):
        self.name = name
        self.network = None
        self.launch_inputs = launch_inputs

        self.sections: list[Section] = [] # the stacking order of sections in the vehicle is the same as the instantiation order

        self.atmospheric_pressure = State(self.launch_inputs.initial_atmospheric_pressure)
        self.length = sum(section.length for section in self.sections)


    def add_network(self, network:Network): # extende to multiple networks in a single vehicle
        if self.network is None:
            self.network = network

    def add_section(self, section:Section): # change to have the order of the sections matter
        if section not in self.sections:
            self.sections.append(section)

    def evaluate(self) -> None:
        self.network.evaluate()
        for section in self.sections:
            section.evaluate()

    def size(self) -> None:
        #for network in self.networks: # eventually, when there are multiple networks
        for component in self.network.components:
            component.size()

    def fly(self, dt: float, t_final: float) -> dict:
        return solve(self, dt, t_final)


    @property
    def mass(self):
        return sum(section.mass for section in self.sections)

    @property
    def length(self):
        return sum(section.length for section in self.sections)

    def __str__(self) -> str:
        lines = [f"Vehicle: {self.name}", f"Sections ({len(self.sections)}):"]
        lines.extend(
            f"  - {section.__class__.__name__}: {section.name}"
            for section in self.sections
        )

        components = self.network.components if self.network is not None else []
        lines.append(f"Components ({len(components)}):")
        lines.extend(
            f"  - {component.__class__.__name__}: {component.name}"
            for component in components
        )
        return "\n".join(lines)
