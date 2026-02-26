from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class NetworkState:
    node: dict
    br: dict

class FluidNetwork:

    def __init__(self, nodes: list, branches: list, params: dict):
        self.nodes = nodes
        self.branches = branches
        self.params = params
        self.state = NetworkState(data={})

    def step_implicit(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # TODO
        return {"network_state": self.state.data}