from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Optional

from .PropSystem import PropSystem
from .flight_forces import gravity
from .types import AeroOut, AtmosState, FluidOut, KinematicsState, PlantOut, ThermalOut


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

    def step_fluids(
        self,
        dt: float,
        atm: AtmosState,
        thermal_out: Optional[ThermalOut] = None,
        commit: bool = True,
    ) -> FluidOut:
        """Advance the fluid network once using the supplied boundary state."""

        heat_flux = (
            thermal_out.heat_flux_to_fluids if thermal_out is not None else {}
        )
        return self.prop_system.update(
            dt=dt,
            atm=atm,
            heat_flux=heat_flux,
            commit=commit,
        )

    def predict_kinematics(
        self,
        kin: KinematicsState,
        plant: PlantOut,
        mass: float,
    ) -> KinematicsState:
        """Explicitly predict the endpoint from beginning-of-step acceleration."""

        dt = kin.dt
        forces = self.forces(kin, plant, mass)
        acceleration = forces["acceleration"]
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

    @staticmethod
    def correct_kinematics(
        kin: KinematicsState,
        start_acceleration: float,
        end_acceleration: float,
        mass: float,
        Ixx: float,
    ) -> KinematicsState:
        """Apply the implicit trapezoidal corrector to the predicted endpoint."""

        dt = kin.dt
        velocity = kin.v + 0.5 * (start_acceleration + end_acceleration) * dt
        altitude = kin.h + 0.5 * (kin.v + velocity) * dt
        return KinematicsState(
            t=kin.t + dt,
            dt=dt,
            h=altitude,
            v=velocity,
            w=kin.w,
            alpha=kin.alpha,
            m=mass,
            Ixx=Ixx,
        )

    @staticmethod
    def forces(
        kin: KinematicsState,
        plant: PlantOut,
        mass: float,
    ) -> Dict[str, float]:
        """Return the forces and acceleration at one synchronized state."""

        thrust = float(plant.fluids.propulsion.thrust)
        drag = math.copysign(float(plant.aero.D), kin.v) if kin.v != 0.0 else 0.0
        weight = gravity(mass, kin.h)
        net = thrust - drag - weight
        return {
            "thrust": thrust,
            "drag": drag,
            "gravity": weight,
            "net": net,
            "acceleration": net / mass,
        }

    def mass_properties(self) -> Dict[str, Any]:
        """Snapshot the vehicle mass state so later updates cannot alter history."""

        return {
            "total_mass": float(self.vehicle.total_mass),
            "cg": float(self.vehicle.cg),
            "Ixx": float(self.vehicle.Ixx),
            "Iyy": float(self.vehicle.Iyy),
            "station": self.vehicle.station.copy(),
            "axial_mass": self.vehicle.mass.copy(),
        }

    def run(self, h0: float = 0.0, v0: float = 0.0) -> List[Dict[str, Any]]:
        """Run the 1D trajectory with an explicit-implicit predictor-corrector."""

        if self.vehicle.total_mass is None:
            self.vehicle.build()

        simulation = self.cfg["simulation"]
        dt = float(simulation["dt"])
        t_end = float(simulation["t_end"])
        corrector_tolerance = float(simulation.get("corrector_tolerance", 1.0e-8))
        corrector_max_iterations = int(simulation.get("corrector_max_iterations", 10))

        atmosphere = self.env.atmosphere(h0, v0)
        fluid_state = self.prop_system.update(
            dt=None,
            atm=atmosphere,
            heat_flux={},
            commit=False,
        )
        self.vehicle.update_mass_distribution(fluid_state.node)
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
            # Evaluate every force used for propagation at the beginning of the
            # interval, using the fluid and vehicle state at the same time.
            kin = replace(
                kin,
                dt=min(dt, t_end - kin.t),
                alpha=self.aero.aoa(kin.t),
                m=float(self.vehicle.total_mass),
                Ixx=float(self.vehicle.Ixx),
            )
            atmosphere = self.env.atmosphere(kin.h, kin.v)
            engine_on = self.powered(fluid_state.propulsion.thrust)
            aero_out = self.aero.evaluate(kin, atmosphere, engine_on)
            thermal_out = self.step_thermal(kin, atmosphere, aero_out)
            start_plant = PlantOut(
                aero=aero_out,
                thermal=thermal_out,
                fluids=fluid_state,
            )
            start_forces = self.forces(
                kin,
                start_plant,
                float(self.vehicle.total_mass),
            )
            predicted_kin = self.predict_kinematics(
                kin,
                start_plant,
                float(self.vehicle.total_mass),
            )

            # Use the explicit predictor to supply endpoint boundary conditions
            # for the single implicit fluid-network propagation.
            predicted_kin = replace(
                predicted_kin,
                alpha=self.aero.aoa(predicted_kin.t),
            )
            predicted_atmosphere = self.env.atmosphere(
                predicted_kin.h,
                predicted_kin.v,
            )
            predicted_aero = self.aero.evaluate(
                predicted_kin,
                predicted_atmosphere,
                engine_on,
            )
            predicted_thermal = self.step_thermal(
                predicted_kin,
                predicted_atmosphere,
                predicted_aero,
            )
            fluid_state = self.step_fluids(
                predicted_kin.dt,
                predicted_atmosphere,
                predicted_thermal,
            )
            self.vehicle.update_mass_distribution(fluid_state.node)

            # Iterate the implicit trapezoidal corrector. Propulsion and mass
            # are fixed at their solved endpoint values; atmosphere and drag
            # are updated with each corrected kinematic state.
            end_engine_on = self.powered(fluid_state.propulsion.thrust)
            next_kin = replace(
                predicted_kin,
                m=float(self.vehicle.total_mass),
                Ixx=float(self.vehicle.Ixx),
            )
            for _ in range(corrector_max_iterations):
                trial_atmosphere = self.env.atmosphere(next_kin.h, next_kin.v)
                trial_aero = self.aero.evaluate(
                    next_kin,
                    trial_atmosphere,
                    end_engine_on,
                )
                trial_plant = PlantOut(
                    aero=trial_aero,
                    thermal=None,
                    fluids=fluid_state,
                )
                end_acceleration = self.forces(
                    next_kin,
                    trial_plant,
                    float(self.vehicle.total_mass),
                )["acceleration"]
                corrected_kin = self.correct_kinematics(
                    kin,
                    start_forces["acceleration"],
                    end_acceleration,
                    float(self.vehicle.total_mass),
                    float(self.vehicle.Ixx),
                )
                corrected_kin = replace(
                    corrected_kin,
                    alpha=self.aero.aoa(corrected_kin.t),
                )
                error = max(
                    abs(corrected_kin.h - next_kin.h)
                    / (1.0 + abs(corrected_kin.h)),
                    abs(corrected_kin.v - next_kin.v)
                    / (1.0 + abs(corrected_kin.v)),
                )
                next_kin = corrected_kin
                if error <= corrector_tolerance:
                    break
            else:
                raise RuntimeError(
                    f"Flight corrector failed to converge at t={next_kin.t:.6g} s"
                )

            # Re-evaluate the complete corrected endpoint for history.
            end_atmosphere = self.env.atmosphere(next_kin.h, next_kin.v)
            end_aero = self.aero.evaluate(next_kin, end_atmosphere, end_engine_on)
            end_thermal = self.step_thermal(next_kin, end_atmosphere, end_aero)
            end_plant = PlantOut(
                aero=end_aero,
                thermal=end_thermal,
                fluids=fluid_state,
            )

            history.append(
                {
                    "kinematics": next_kin,
                    "atmosphere": end_atmosphere,
                    "plant": end_plant,
                    "forces": self.forces(
                        next_kin,
                        end_plant,
                        float(self.vehicle.total_mass),
                    ),
                    "mass_properties": self.mass_properties(),
                    "engine_on": end_engine_on,
                }
            )
            kin = next_kin

        return history

    def on_rail(self, altitude: float) -> bool:
        launch = self.cfg["launch"]
        return altitude < float(launch["altitude"]) + float(launch["rail_length"])

    @staticmethod
    def powered(thrust: float) -> bool:
        return thrust > 0.0
