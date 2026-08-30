from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Component import Component
    from .State import State

class Network:

    def __init__(self, name: str):
        self.name = name
        self.components: list[Component] = []

    def add_component(self, component: Component) -> None:
        if component not in self.components:
            self.components.append(component)



    def collect_states(self) -> list[State]:
        states = []
        for component in self.components:
            for dynamic in component.dynamics:
                state = dynamic[0]
                if state not in states:
                    states.append(state)
        for component in self.components:
            for state, _ in component.balances:
                if state not in states:
                    states.append(state)
        return states



    def collect_stored_states(self) -> list[State]:
        states = []
        for component in self.components:
            for dynamic in component.dynamics:
                if len(dynamic) == 2:
                    state, _ = dynamic
                    stored_state = state
                elif len(dynamic) == 3:
                    _, stored_state, _ = dynamic
                else:
                    raise ValueError("Dynamics must contain 2 or 3 values.")
                states.append(stored_state)
        return states



    def collect_derivatives(self) -> list[float]:
        derivatives = []
        for component in self.components:
            for dynamic in component.dynamics:
                if len(dynamic) == 2:
                    _, derivative = dynamic
                elif len(dynamic) == 3:
                    _, _, derivative = dynamic
                else:
                    raise ValueError("Dynamics must contain 2 or 3 values.")
                derivatives.append(derivative)
        return derivatives



    def collect_balances(self) -> list[float]:
        balances = []
        for component in self.components:
            for _, balance in component.balances:
                balances.append(balance)
        return balances



    def evaluate(self) -> None:
        for c in self.components:
            c.evaluate()


    def __str__(self) -> str:
        lines = [
            f"Network: {self.name}",
            f"Components ({len(self.components)}):",
        ]
        lines.extend(
            f"  ├─ [{component.__class__.__name__}]: {component.name}"
            for component in self.components
        )
        return "\n".join(lines)