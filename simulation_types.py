"""Dependency-neutral state/output contracts shared by flight, fluids and thermals."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from constraints import ConstraintRecord, feasibility


@dataclass
class KinematicsState:
    t: float
    dt: float
    h: float
    v: float
    w: float
    alpha: float
    m: float
    Ixx: float


@dataclass
class AtmosState:
    T: float
    p: float
    rho: float
    mu: float
    a: float
    q: float
    Ma: float


@dataclass
class AeroOut:
    Cd: float
    D: float
    Ca: float = 0.0
    A: float = 0.0
    Cn: float = 0.0
    N: float = 0.0
    cp: float = float("nan")
    Mroll: float = 0.0


@dataclass
class ThermalOut:
    node: Dict[str, Dict[str, Any]]

    def heat_rates(self) -> Dict[str, Dict[str, float]]:
        """Return phase heat rates in watts for fluid-network nodes."""

        return {
            output.get("fluid_node_id", node_id): {
                phase: float(boundary["heat_rate"])
                for phase, boundary in output.get("phases", {}).items()
            }
            for node_id, output in self.node.items()
            if output.get("phases")
        }


@dataclass
class PropulsionOut:
    mode: str
    shutdown_reason: Optional[str]
    thrust: float
    Pc: float
    MR: float
    Cf: float
    cstar: float
    mdot_ox: float
    mdot_fuel: float
    mdot_nozzle: float


@dataclass
class FluidOut:
    node: Dict[str, Dict[str, Any]]
    branch: Dict[str, Dict[str, Any]]
    td_state: Dict[str, Dict[str, Any]]
    mdot: Dict[str, float]
    propulsion: PropulsionOut
    events: tuple[Dict[str, Any], ...] = ()
    event_counts: Dict[str, int] = field(default_factory=dict)
    constraints: Dict[str, float] = field(default_factory=dict)
    constraint_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class PlantOut:
    aero: AeroOut
    thermal: Optional[ThermalOut]
    fluids: FluidOut


@dataclass
class SimResult:
    """Scalar optimizer output; SI units, stability in body diameters.

    Extrema are sampled at flight endpoints. Apogee uses interpolation of the
    first upward-to-downward velocity crossing, not event-split integration.
    A missing apogee or incomplete burn must not be treated as an achieved target.
    Unavailable masses on preflight rejection are None. Use accepted, not
    feasible alone, to select completed designs that meet the mission limits.
    """
    max_altitude: float = 0.0
    apogee: Optional[float] = None
    apogee_time: Optional[float] = None
    initial_mass: Optional[float] = None
    dry_mass: Optional[float] = None
    final_mass: Optional[float] = None
    max_q: float = 0.0
    min_stability_calibers: Optional[float] = None
    burn_duration: float = 0.0
    burn_complete: bool = False
    final_time: float = 0.0
    final_altitude: float = 0.0
    final_velocity: float = 0.0
    termination: str = "time_limit"
    geometry_constraints: Dict[str, Optional[float]] = field(default_factory=dict)
    constraints: Dict[str, Optional[float]] = field(default_factory=dict)
    history: Optional[list] = None
    pump_sizing: Dict[str, Dict[str, float]] = field(default_factory=dict)
    constraint_records: Dict[str, ConstraintRecord] = field(default_factory=dict)
    constraint_times: Dict[str, float] = field(default_factory=dict)
    min_rail_twr: Optional[float] = None
    min_rail_twr_time: Optional[float] = None
    rail_exit_time: Optional[float] = None
    max_twr: float = 0.0
    load_peaks: Dict[str, float] = field(default_factory=dict)
    shutdown_reason: Optional[str] = None

    @property
    def completed(self) -> bool:
        return self.apogee_reached

    @property
    def accepted(self) -> bool:
        if not self.completed or not self.burn_complete:
            return False
        if not {"goal_apogee", "max_burn_duration"} <= self.constraints.keys():
            return False
        if self.constraint_records:
            return feasibility({k: r.margin for k, r in self.constraint_records.items() if r.required}) is True
        return self.feasible is True

    @property
    def apogee_reached(self) -> bool:
        return self.apogee is not None

    @property
    def feasible(self) -> Optional[bool]:
        """False for any violation; None when any requested check is unassessed."""
        return feasibility({**self.geometry_constraints, **self.constraints})
