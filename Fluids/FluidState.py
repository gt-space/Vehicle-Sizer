"""Evaluated fluid and component records.

Records contain data, connectivity, and solver history. 

"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FluidState(Mapping):
    """One fluid's identity, represented phase and evaluated properties."""

    fluid: str
    phase: str
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if {"fluid", "phase"} & self.properties.keys():
            raise ValueError("Fluid identity and phase belong in their explicit fields")
        self.properties = dict(self.properties)

    @classmethod
    def from_dict(cls, fluid, values):
        """Read a detached fluid report; fluid identity is its enclosing key."""
        values = deepcopy(dict(values))
        if values.pop("fluid", fluid) != fluid:
            raise ValueError("Fluid record identity does not match its key")
        return cls(fluid, values.pop("phase", "unknown"), values)

    def as_dict(self):
        # Existing reports carry identity in the enclosing key or fluid_name.
        return deepcopy({**self.properties, "phase": self.phase})

    def __getitem__(self, key):
        if key == "fluid":
            return self.fluid
        if key == "phase":
            return self.phase
        return self.properties[key]

    def __iter__(self):
        yield "fluid"
        yield "phase"
        yield from self.properties

    def __len__(self):
        return len(self.properties) + 2


@dataclass
class NodeState(Mapping):
    """One evaluated node, including its constituent fluid records.

    trial_values contains exactly the local solver unknowns. properties holds
    derived/prescribed quantities. evaluation_data carries intermediates needed
    by residuals/events and is excluded from mapping access and public reports.
    """

    trial_values: dict[str, float] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    fluids: dict[str, FluidState] = field(default_factory=dict)
    evaluation_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if (self.trial_values.keys() & self.properties.keys()
                or "fluids" in self.trial_values or "fluids" in self.properties):
            raise ValueError("Node trial values and properties must have distinct, non-reserved names")
        self.trial_values = dict(self.trial_values)
        self.properties = dict(self.properties)
        self.fluids = dict(self.fluids)
        self.evaluation_data = dict(self.evaluation_data)

    @classmethod
    def from_dict(cls, values):
        """Read boundary/report data, placing quantities in properties.

        A flat report cannot identify solver unknowns or restore evaluation
        data. Build trial records explicitly instead of using this method.
        """
        values = deepcopy(dict(values))
        fluids = {name: fluid if isinstance(fluid, FluidState)
                  else FluidState.from_dict(name, fluid)
                  for name, fluid in values.pop("fluids", {}).items()}
        return cls(properties=values, fluids=fluids)

    def as_dict(self):
        values = deepcopy({**self.trial_values, **self.properties})
        if self.fluids:
            values["fluids"] = {name: fluid.as_dict() for name, fluid in self.fluids.items()}
        return values

    def __getitem__(self, key):
        if key == "fluids":
            return self.fluids
        if key in self.trial_values:
            return self.trial_values[key]
        return self.properties[key]

    def __iter__(self):
        yield from self.trial_values
        yield from self.properties
        yield "fluids"

    def __len__(self):
        return len(self.trial_values) + len(self.properties) + 1


@dataclass
class BranchState(Mapping):
    """One evaluated branch's unknowns and transported streams.

    Each flow holds signed mdot, a fixed donor direction, and a FluidState.
    mdot/phase_mdot sum transported flows; trial_values retains the raw solver
    guess even for a closed path. Mapping access reads those raw unknowns.
    """

    trial_values: dict[str, float] = field(default_factory=dict)
    flows: dict[str, dict[str, Any]] = field(default_factory=dict)
    dP: float = 0.0
    enabled: bool = True
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        reserved = {"dP", "enabled", "flows"}
        if (self.trial_values.keys() & self.properties.keys()
                or reserved & (self.trial_values.keys() | self.properties.keys())):
            raise ValueError("Branch trial values and properties must have distinct, non-reserved names")
        self.trial_values = dict(self.trial_values)
        self.properties = dict(self.properties)
        self.flows = {name: dict(flow) for name, flow in self.flows.items()}

    @property
    def mdot(self):
        return sum(float(flow["mdot"]) for flow in self.flows.values()) if self.enabled else 0.0

    def phase_mdot(self, phase):
        return sum(float(flow["mdot"]) for flow in self.flows.values()
                   if flow["fluid"].phase == phase) if self.enabled else 0.0

    def as_dict(self):
        return deepcopy({
            **self.trial_values,
            "dP": self.dP,
            "enabled": self.enabled,
            "flows": {
                name: {**flow, "fluid": flow["fluid"].as_dict(),
                       "fluid_name": flow["fluid"].fluid}
                for name, flow in self.flows.items()
            },
            **self.properties,
        })

    def __getitem__(self, key):
        if key in {"flows", "dP", "enabled"}:
            return getattr(self, key)
        if key in self.trial_values:
            return self.trial_values[key]
        return self.properties[key]

    def __iter__(self):
        yield from self.trial_values
        yield "dP"
        yield "enabled"
        yield "flows"
        yield from self.properties

    def __len__(self):
        return len(self.trial_values) + len(self.properties) + 3


__all__ = ["FluidState", "NodeState", "BranchState"]
