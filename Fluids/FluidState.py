from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator


@dataclass
class FluidState(MutableMapping[str, Any]):
    """Thermodynamic state carried by a node or branch flow."""

    fluid: str
    phase: str
    properties: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, fluid: str, values: Dict[str, Any]) -> "FluidState":
        values = dict(values)
        return cls(fluid, str(values.pop("phase", "unknown")), values)

    def as_dict(self) -> Dict[str, Any]:
        return {**self.properties, "phase": self.phase}

    def __getitem__(self, key: str) -> Any:
        if key == "fluid":
            return self.fluid
        if key == "phase":
            return self.phase
        return self.properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "fluid":
            self.fluid = str(value)
        elif key == "phase":
            self.phase = str(value)
        else:
            self.properties[key] = value

    def __delitem__(self, key: str) -> None:
        if key in ("fluid", "phase"):
            raise KeyError(key)
        del self.properties[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.properties
        yield "phase"

    def __len__(self) -> int:
        return len(self.properties) + 1


@dataclass
class NodeState(MutableMapping[str, Any]):
    """General state owned by a fluid node."""

    state: Dict[str, Any] = field(default_factory=dict)
    fluids: Dict[str, FluidState] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "NodeState":
        values = dict(values)
        fluids = {
            name: value if isinstance(value, FluidState) else FluidState.from_dict(name, value)
            for name, value in values.pop("fluids", {}).items()
        }
        return cls(values, fluids)

    def as_dict(self) -> Dict[str, Any]:
        values = dict(self.state)
        if self.fluids:
            values["fluids"] = {
                name: fluid.as_dict() for name, fluid in self.fluids.items()
            }
        return values

    def __getitem__(self, key: str) -> Any:
        if key == "fluids":
            return self.fluids
        return self.state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "fluids":
            self.fluids = value
        else:
            self.state[key] = value

    def __delitem__(self, key: str) -> None:
        if key == "fluids":
            self.fluids.clear()
        else:
            del self.state[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.state
        if self.fluids:
            yield "fluids"

    def __len__(self) -> int:
        return len(self.state) + bool(self.fluids)


@dataclass
class BranchState(MutableMapping[str, Any]):
    """General state and transported flows owned by a fluid branch."""

    state: Dict[str, Any] = field(default_factory=dict)
    flows: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dP: float = 0.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def mdot(self) -> float:
        return sum(float(flow["mdot"]) for flow in self.flows.values())

    def phase_mdot(self, phase: str) -> float:
        return sum(
            float(flow["mdot"])
            for flow in self.flows.values()
            if flow["fluid"].phase == phase
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            **self.state,
            "dP": self.dP,
            "enabled": self.enabled,
            "flows": {
                name: {
                    **{key: value for key, value in flow.items() if key != "fluid"},
                    "fluid": flow["fluid"].as_dict(),
                    "fluid_name": flow["fluid"].fluid,
                }
                for name, flow in self.flows.items()
            },
            **self.metadata,
        }

    def __getitem__(self, key: str) -> Any:
        if key == "flows":
            return self.flows
        if key == "dP":
            return self.dP
        if key == "enabled":
            return self.enabled
        if key == "mdot":
            return self.mdot
        if key in self.state:
            return self.state[key]
        return self.metadata[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "flows":
            self.flows = value
        elif key == "dP":
            self.dP = float(value)
        elif key == "enabled":
            self.enabled = bool(value)
        elif key == "mdot":
            if len(self.flows) != 1:
                raise KeyError("Set a named flow when a branch has multiple flows")
            next(iter(self.flows.values()))["mdot"] = float(value)
        else:
            self.state[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self.state:
            del self.state[key]
        elif key in self.metadata:
            del self.metadata[key]
        else:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield from self.state
        yield "dP"
        yield "enabled"
        yield "flows"
        yield from self.metadata

    def __len__(self) -> int:
        return len(self.state) + len(self.metadata) + 3
