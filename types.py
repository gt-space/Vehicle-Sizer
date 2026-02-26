from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Protocol, Optional

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
    Tr: float

@dataclass
class AeroOut:
    D: float
    heat_bc: Dict[str, Any]
    Mroll: float = 0.0

@dataclass
class ThermalOut:
    wall_T: Any
    Qdot_to_fluids: Dict[str, float]

@dataclass
class FluidOut:
    thrust: float
    dm_fuel: float
    dm_ox: float
    dm_press: float

    Pc: float
    Pox: float
    Pfuel: float
    Pcopv: float

    Pvec: Tuple[float, float, float]

@dataclass
class PlantOut:
    aero: AeroOut
    thermal: ThermalOut
    fluids: FluidOut