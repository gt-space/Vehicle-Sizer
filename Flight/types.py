from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional


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
    heat_bc: Dict[str, Any]
    Mroll: float = 0.0


@dataclass
class ThermalOut:
    wall_T: Any
    heat_flux_to_fluids: Dict[str, float]


@dataclass
class PropulsionOut:
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


@dataclass
class PlantOut:
    aero: AeroOut
    thermal: Optional[ThermalOut]
    fluids: FluidOut
