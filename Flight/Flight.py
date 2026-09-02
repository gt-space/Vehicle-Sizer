from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .PropSystem import PropSystem
from .flight_forces import gravity
from .types import AeroOut, AtmosState, KinematicsState, PlantOut, ThermalOut


class FlightSim:
    """Coordinate atmosphere, propulsion, vehicle mass, and 1D kinematics."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        env: Any,
        aero: Any,
        prop_system: PropSystem,
        vehicle: Any,
        thermal: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.aero = aero
        self.prop_system = prop_system
        self.vehicle = vehicle
        self.thermal = thermal

    def step_thermal(
        self,
        kin: KinematicsState,
        atm: AtmosState,
        aero_out: AeroOut,
    ) -> Optional[ThermalOut]:
        """Extension point for the wall-to-fluid heat-transfer solve."""

        if self.thermal is None:
            return None
        raise NotImplementedError("Thermal coupling has not been connected")

    def step_plant(
        self,
        kin: KinematicsState,
        atm: AtmosState,
        aero_out: AeroOut,
        thermal_out: Optional[ThermalOut] = None,
        commit: bool = True,
    ) -> PlantOut:
        """Advance the propulsion system once using the current atmosphere."""

        heat_flux = (
            thermal_out.heat_flux_to_fluids if thermal_out is not None else {}
        )
        fluid_out = self.prop_system.update(
            dt=kin.dt,
            atm=atm,
            heat_flux=heat_flux,
            commit=commit,
        )
        return PlantOut(aero=aero_out, thermal=thermal_out, fluids=fluid_out)

    def step_kinematics(
        self,
        kin: KinematicsState,
        plant: PlantOut,
        mass: float,
    ) -> KinematicsState:
        """Propagate vertical position and velocity with constant acceleration."""

        if mass <= 0.0:
            raise ValueError("Vehicle mass must remain positive")

        dt = kin.dt
        thrust = float(plant.fluids["propulsion"]["thrust"])
        drag = math.copysign(float(plant.aero.D), kin.v) if kin.v != 0.0 else 0.0
        acceleration = (thrust - drag - gravity(mass, kin.h)) / mass
        velocity = kin.v + acceleration * dt
        altitude = kin.h + kin.v * dt + 0.5 * acceleration * dt**2

        return KinematicsState(
            t=kin.t + dt,
            dt=dt,
            h=altitude,
            v=velocity,
            w=kin.w,
            alpha=kin.alpha,
            m=mass,
            Ixx=kin.Ixx,
        )

    def run(self, h0: float = 0.0, v0: float = 0.0) -> List[Dict[str, Any]]:
        """Run the 1D trajectory until the configured end time or apogee."""

        if self.vehicle.total_mass is None:
            self.vehicle.build()

        simulation = self.cfg["simulation"]
        dt = float(simulation["dt"])
        t_end = float(simulation["t_end"])
        atmosphere = self.env.atmosphere(h0, v0)
        initial_fluid = self.prop_system.update(
            dt=None,
            atm=atmosphere,
            heat_flux={},
            commit=False,
        )
        self.vehicle.update_mass_distribution(initial_fluid["node"])
        kin = KinematicsState(
            t=0.0,
            dt=dt,
            h=h0,
            v=v0,
            w=0.0,
            alpha=0.0,
            m=float(self.vehicle.total_mass),
            Ixx=float(self.vehicle.Ixx),
        )
        history: List[Dict[str, Any]] = []

        while kin.t < t_end and (kin.t == 0.0 or kin.v >= 0.0):
            atmosphere = self.env.atmosphere(kin.h, kin.v)
            kin.alpha = self.aero.aoa(kin, atmosphere, self.on_rail(kin.h))
            aero_out = self.aero.evaluate(kin, atmosphere)
            thermal_out = self.step_thermal(kin, atmosphere, aero_out)
            plant = self.step_plant(kin, atmosphere, aero_out, thermal_out)

            self.vehicle.update_mass_distribution(plant.fluids["node"])
            kin.Ixx = float(self.vehicle.Ixx)
            kin = self.step_kinematics(kin, plant, float(self.vehicle.total_mass))
            history.append(
                {
                    "kinematics": kin,
                    "atmosphere": atmosphere,
                    "plant": plant,
                }
            )

        return history

    def on_rail(self, altitude: float) -> bool:
        launch = self.cfg["launch"]
        return altitude < float(launch["altitude"]) + float(launch["rail_length"])

    @staticmethod
    def powered(thrust: float) -> bool:
        return thrust > 0.0
