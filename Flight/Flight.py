from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np

from Fluids.PropSystem import PropSystem
from .flight_forces import gravity
from .loads import Loads
from simulation_types import SimResult
from constraints import merge_margins
from simulation_types import (
    AeroOut,
    AtmosState,
    FluidOut,
    KinematicsState,
    PlantOut,
    PropulsionOut,
    ThermalOut,
)


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
        self.loads = Loads(vehicle, aero)

    def trial_aero(
        self,
        kin: KinematicsState,
        atm: AtmosState,
        engine_on: bool,
    ) -> AeroOut:
        return self.aero.evaluate(kin, atm, engine_on)

    def trial_thermal(
        self,
        kin: KinematicsState,
        atm: AtmosState,
        aero_out: AeroOut,
        fluids: FluidOut,
        previous: Optional[ThermalOut] = None,
        axial_specific_force: float = 0.0,
    ) -> Optional[ThermalOut]:
        """Evaluate wall temperatures without committing thermal state."""

        if self.thermal is None:
            return None
        return self.thermal.trial(
            kin,
            atm,
            aero_out,
            fluids,
            previous,
            axial_specific_force=axial_specific_force,
        )

    def trial_fluid(
        self,
        dt: Optional[float],
        atm: AtmosState,
        thermal_out: Optional[ThermalOut] = None,
        axial_specific_force: float = 0.0,
    ) -> FluidOut:
        """Evaluate fluid state without committing it."""

        return self.prop_system.update(
            dt=dt,
            atm=atm,
            heat_rate=thermal_out.heat_rates() if thermal_out is not None else {},
            commit=False,
            axial_specific_force=axial_specific_force,
        )

    def commit_fluid(
        self,
        dt: Optional[float],
        atm: AtmosState,
        thermal_out: Optional[ThermalOut] = None,
        axial_specific_force: float = 0.0,
    ) -> FluidOut:
        """Advance and commit the fluid network with converged heat rates."""

        return self.prop_system.update(
            dt=dt,
            atm=atm,
            heat_rate=thermal_out.heat_rates() if thermal_out is not None else {},
            commit=True,
            axial_specific_force=axial_specific_force,
        )

    @staticmethod
    def heat_rate_error(previous: ThermalOut, current: ThermalOut) -> float:
        def flatten(output: ThermalOut) -> Dict[tuple, float]:
            return {
                (node_id, phase): value
                for node_id, phases in output.heat_rates().items()
                for phase, value in phases.items()
            }

        old = flatten(previous)
        new = flatten(current)
        return max(
            (
                abs(new.get(key, 0.0) - old.get(key, 0.0))
                / (1.0 + abs(new.get(key, 0.0)))
                for key in old.keys() | new.keys()
            ),
            default=0.0,
        )

    @staticmethod
    def wall_temperature_error(previous: ThermalOut, current: ThermalOut) -> float:
        return max(
            (
                float(np.max(np.abs(
                    current.node[node_id]["cells"]["wall_T"]
                    - output["cells"]["wall_T"]
                )))
                for node_id, output in previous.node.items()
            ),
            default=0.0,
        )

    def step_coupled(
        self,
        kin: KinematicsState,
        atm: AtmosState,
        aero_out: AeroOut,
        fluids: FluidOut,
        axial_specific_force: float,
    ) -> tuple[FluidOut, Optional[ThermalOut]]:
        """Converge external heat boundaries, then commit the fluid step once."""

        if self.thermal is None:
            return (
                self.commit_fluid(
                    kin.dt,
                    atm,
                    axial_specific_force=axial_specific_force,
                ),
                None,
            )

        settings = self.cfg.get("advanced", {}).get("heating", {})
        tolerance = float(settings.get("convergence_tolerance", 1.0e-3))
        temperature_tolerance = float(settings.get("temperature_tolerance", 1.0e-2))
        max_iterations = int(settings.get("max_iterations", 5))
        if min(tolerance, temperature_tolerance) < 0.0:
            raise ValueError("Heating convergence tolerances must be nonnegative")
        if max_iterations < 1:
            raise ValueError("Heating convergence iterations must be positive")

        thermal_out = self.trial_thermal(
            kin,
            atm,
            aero_out,
            fluids,
            axial_specific_force=axial_specific_force,
        )
        if thermal_out is None:
            raise RuntimeError("Thermal model did not return boundary conditions")
        for _ in range(max_iterations):
            trial_fluids = self.trial_fluid(
                kin.dt,
                atm,
                thermal_out=thermal_out,
                axial_specific_force=axial_specific_force,
            )
            next_thermal = self.trial_thermal(
                kin,
                atm,
                aero_out,
                trial_fluids,
                previous=thermal_out,
                axial_specific_force=axial_specific_force,
            )
            if next_thermal is None:
                raise RuntimeError("Thermal model did not return boundary conditions")
            if (
                self.heat_rate_error(thermal_out, next_thermal) <= tolerance
                and self.wall_temperature_error(thermal_out, next_thermal)
                <= temperature_tolerance
            ):
                thermal_out = next_thermal
                break
            thermal_out = next_thermal
        else:
            raise RuntimeError("Fluid/thermal coupling failed to converge")

        fluids = self.commit_fluid(
            kin.dt,
            atm,
            thermal_out=thermal_out,
            axial_specific_force=axial_specific_force,
        )
        self.thermal.commit(thermal_out)
        return fluids, thermal_out


    def predict_kinematics(
        self,
        kin: KinematicsState,
        plant: PlantOut,
        mass: float,
    ) -> KinematicsState:
        """Explicitly predict the endpoint from beginning-of-step acceleration."""

        dt = kin.dt
        forces = self.forces(kin, plant, mass)

        ax = forces["ax"]
        az = forces["az"]
        q_dot = forces["pitch_acceleration"]

        vx = kin.vx + ax * dt
        vz = kin.vz + az * dt

        x = kin.x + kin.vx * dt + 0.5 * ax * dt**2
        h = kin.h + kin.vz * dt + 0.5 * az * dt**2

        q = kin.q + q_dot * dt
        theta = kin.theta + kin.q * dt + 0.5 * q_dot * dt**2

        return KinematicsState(
            t=kin.t + dt,
            dt=dt,
            x=x,
            h=h,
            vx=vx,
            vz=vz,
            theta=theta,
            q=q,
            alpha=kin.alpha,
            m=mass,
            Iyy=kin.Iyy,
        )


    @staticmethod
    def correct_kinematics(
        kin: KinematicsState,
        start_acceleration: float,
        end_acceleration: float,
        mass: float,
        Iyy: float,
    ) -> KinematicsState:
        """Apply the implicit trapezoidal corrector to the predicted endpoint."""

        dt = kin.dt
        velocity = kin.vz + 0.5 * (start_acceleration + end_acceleration) * dt
        altitude = kin.h + 0.5 * (kin.vz + velocity) * dt
        return KinematicsState(
            t=kin.t + dt,
            dt=dt,
            x=kin.x,
            h=altitude,
            vx=kin.vx,
            vz=velocity,
            theta=kin.theta,
            q=kin.q,
            alpha=kin.alpha,
            m=mass,
            Iyy=Iyy,
        )

    def forces(
        self,
        kin: KinematicsState,
        plant: PlantOut,
        mass: float,
    ) -> Dict[str, float]:
        """
        Forces, moments, and state derivatives needed to update the 3DOF state vector.
        From this, only the accelerations and pitch rate outputs really matter.
        """

        thrust = float(plant.fluids.propulsion.thrust)
        axial_aero = float(plant.aero.A)
        normal_aero = float(plant.aero.N)
        cp = float(plant.aero.cp)
        cg = float(self.vehicle.cg)
        weight = gravity(mass, kin.h)

        if (not all(math.isfinite(x) for x in (
                    thrust,
                    axial_aero,
                    normal_aero,
                    cp,
                    cg,
                    weight,
                    mass,
                    kin.h,
                    kin.theta,
                    kin.Iyy,
                )
            ) or mass <= 0.0 or weight <= 0.0 or kin.Iyy <= 0.0
        ):
            raise ValueError("Nonphysical or nonfinite flight force state")

        Fx = (thrust - axial_aero) * math.cos(kin.theta) - normal_aero * math.sin(kin.theta)
        Fz = (thrust - axial_aero) * math.sin(kin.theta) + normal_aero * math.cos(kin.theta) - weight

        ax = Fx / mass
        az = Fz / mass

        pitch_moment = -normal_aero * (cp - cg)
        pitch_acceleration = pitch_moment / kin.Iyy

        return {
            "thrust": thrust,
            "axial_aero": axial_aero,
            "normal_aero": normal_aero,
            "gravity": weight,
            "Fx": Fx,
            "Fz": Fz,
            "ax": ax,
            "az": az,
            "pitch_moment": pitch_moment,
            "pitch_acceleration": pitch_acceleration,
            "twr": thrust / weight,
            "drag": axial_aero,
            "net": Fz,
            "acceleration": az,
        }



    def mass_properties(self) -> Dict[str, Any]:
        """Snapshot the vehicle mass state so later updates cannot alter history."""

        return {
            "total_mass": float(self.vehicle.total_mass),
            "cg": float(self.vehicle.cg),
            "Ixx": float(self.vehicle.Ixx),
            "Iyy": float(self.vehicle.Iyy),
            "length": float(self.vehicle.length),
            "station": self.vehicle.station.copy(),
            "axial_mass": self.vehicle.mass.copy(),
        }

    @staticmethod
    def stop_fluids(fluids: FluidOut) -> FluidOut:
        """Freeze inventories and report closed, zero-flow feed paths."""

        return replace(
            fluids,
            mdot={branch_id: 0.0 for branch_id in fluids.mdot},
            propulsion=replace(
                fluids.propulsion,
                thrust=0.0,
                mdot_ox=0.0,
                mdot_fuel=0.0,
                mdot_nozzle=0.0,
            ),
        )

    @staticmethod
    def flight_kinematics(
        kin: KinematicsState,
        wind_x: float = 0.0,
        wind_z: float = 0.0,
    ):
        speed = math.hypot(kin.vx, kin.vz)

        # flight-path angle
        gamma = math.atan2(kin.vz, kin.vx) if speed > 1.0e-8 else kin.theta # if vehicle is on rail

        vx_air = kin.vx - wind_x
        vz_air = kin.vz - wind_z

        airspeed = math.hypot(vx_air, vz_air)

        if airspeed > 1.0e-8:
            gamma_air = math.atan2(vz_air, vx_air)
            alpha = kin.theta - gamma_air # angle of attack (wind-relative)
        else:
            gamma_air = kin.theta # flight-path relative to wind 
            alpha = 0.0 # on rail

        return speed, gamma, airspeed, gamma_air, alpha


    def run(self, h0: float = 0.0, v0: float = 0.0, *, progress=None,
            record_history: bool = True, compute_loads: bool = True) -> List[Dict[str, Any]]:
        """Run the 1D trajectory with an explicit-implicit predictor-corrector."""

        if self.vehicle.total_mass is None:
            raise RuntimeError("Vehicle must be built before starting FlightSim")
        mass = float(self.vehicle.total_mass)

        simulation = self.cfg["simulation"]
        dt = float(simulation["dt"])
        t_end = float(simulation["t_end"])
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(t_end) or t_end < 0:
            raise ValueError("Simulation requires finite dt > 0 and t_end >= 0")
        fluid_solve_post_shutdown = simulation.get(
            "fluid_solve_post_shutdown", True
        )
        if not isinstance(fluid_solve_post_shutdown, bool):
            raise ValueError("simulation.fluid_solve_post_shutdown must be boolean")
        advanced = self.cfg.get("advanced", {}).get("flight", simulation)
        corrector_tolerance = float(advanced.get("corrector_tolerance", 1.0e-8))
        corrector_max_iterations = int(advanced.get("corrector_max_iterations", 10))

        atmosphere = self.env.atmosphere(h0, abs(v0))
        fluid_state = self.commit_fluid(
            dt=None,
            atm=atmosphere,
            axial_specific_force=gravity(mass, h0) / mass,
        )
        self.vehicle.update_mass_distribution(fluid_state.node)
        mass = float(self.vehicle.total_mass)
        inertia = float(self.vehicle.Iyy)

        # initial condition definition
        # might be good to eventually put these in the config so that they're not hardcoded
        # also could add automatic initial condition calcs from rail height and angle.
        kin = KinematicsState(
            t=0.0,
            dt=dt,
            x=0.0, 
            h=h0,
            vx=0.0,
            vz=v0,
            theta=np.pi / 2,
            q=0.0,
            alpha=0.0,
            m=mass,
            Iyy=inertia,
        )
        history: List[Dict[str, Any]] = []
        result = self.result = SimResult(
            max_altitude=h0, initial_mass=mass, final_mass=mass,
            dry_mass=float(np.sum(getattr(self.vehicle, "dry_mass", self.vehicle.mass))),
            final_altitude=h0, final_velocity=v0,
            geometry_constraints=dict(getattr(self.vehicle, "geometry_constraints", {})),
            history=history if record_history else None,
            constraints=dict(fluid_state.constraints),
            constraint_times=dict(fluid_state.constraint_times),
        )
        thermal_out = None
        result.burn_complete = fluid_state.propulsion.mode == "shutdown"
        result.shutdown_reason = fluid_state.propulsion.shutdown_reason
        initial_aero = self.trial_aero(kin, atmosphere, self.engine_on(fluid_state.propulsion))
        initial_forces = self.forces(kin, PlantOut(initial_aero, None, fluid_state), mass)
        self._record_twr(result, kin, initial_forces["twr"])
        if compute_loads:
            self._evaluate_loads(result, kin, atmosphere, initial_aero, initial_forces,
                                 self.engine_on(fluid_state.propulsion))
        if progress is not None:
            progress(kin)

        while kin.t < t_end and (kin.t == 0.0 or kin.vz >= 0.0):
            # Evaluate every force used for propagation at the beginning of the
            # interval, using the fluid and vehicle state at the same time.
            mass = float(self.vehicle.total_mass)
            inertia = float(self.vehicle.Iyy)
            kin = replace(
                kin,
                dt=min(dt, t_end - kin.t),
                m=mass,
                Iyy=inertia,
            )

            _, _, airspeed, _, alpha = self.flight_kinematics(kin)
            if self.on_rail(kin.h):
                alpha = 0.0
            kin = replace(kin, alpha=alpha)
            atmosphere = self.env.atmosphere(kin.h, airspeed)

            engine_on = self.engine_on(fluid_state.propulsion)
            aero_out = self.trial_aero(kin, atmosphere, engine_on)
            self._record_extrema(result, atmosphere, aero_out)
            start_plant = PlantOut(
                aero=aero_out,
                thermal=None,
                fluids=fluid_state,
            )
            start_forces = self.forces(kin, start_plant, mass)
            predicted_kin = self.predict_kinematics(
                kin,
                start_plant,
                mass,
            )

            # Use the explicit predictor to supply endpoint boundary conditions
            # for the single implicit fluid-network propagation.
            _, _, predicted_airspeed, _, predicted_alpha = self.flight_kinematics(predicted_kin)
            if self.on_rail(predicted_kin.h):
                predicted_alpha = 0.0
            predicted_kin = replace(predicted_kin, alpha=predicted_alpha)
            predicted_atmosphere = self.env.atmosphere(
                predicted_kin.h,
                predicted_airspeed,
            )


            predicted_aero = self.trial_aero(
                predicted_kin,
                predicted_atmosphere,
                engine_on,
            )
            if fluid_solve_post_shutdown or fluid_state.propulsion.mode != "shutdown":
                try:
                    fluid_state, thermal_out = self.step_coupled(
                        predicted_kin,
                        predicted_atmosphere,
                        predicted_aero,
                        fluid_state,
                        axial_specific_force=(
                            start_forces["thrust"] - start_forces["drag"]
                        )
                        / mass,
                    )
                except RuntimeError as error:
                    raise RuntimeError(
                        f"Fluid solve failed from t={kin.t:.6g} s to "
                        f"t={predicted_kin.t:.6g} s"
                    ) from error
                if (
                    not fluid_solve_post_shutdown
                    and fluid_state.propulsion.mode == "shutdown"
                ):
                    fluid_state = self.stop_fluids(fluid_state)
            else:
                fluid_state = replace(fluid_state, events=())
            self.vehicle.update_mass_distribution(fluid_state.node)
            mass = float(self.vehicle.total_mass)
            inertia = float(self.vehicle.Iyy)

            # Iterate the implicit trapezoidal corrector. Propulsion and mass
            # are fixed at their solved endpoint values; atmosphere and drag
            # are updated with each corrected kinematic state.
            end_engine_on = self.engine_on(fluid_state.propulsion)
            next_kin = replace(
                predicted_kin,
                m=mass,
                Iyy=inertia,
            )
            for _ in range(corrector_max_iterations):

                _, _, trial_airspeed, _, _ = self.flight_kinematics(next_kin)
                trial_atmosphere = self.env.atmosphere(next_kin.h, trial_airspeed)

                trial_aero = self.trial_aero(
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
                    mass,
                )["acceleration"]
                corrected_kin = self.correct_kinematics(
                    kin,
                    start_forces["acceleration"],
                    end_acceleration,
                    mass,
                    inertia,
                )
                _, _, _, _, corrected_alpha = self.flight_kinematics(corrected_kin)
                if self.on_rail(corrected_kin.h):
                    corrected_alpha = 0.0
                corrected_kin = replace(corrected_kin, alpha=corrected_alpha)
                error = max(
                    abs(corrected_kin.h - next_kin.h)
                    / (1.0 + abs(corrected_kin.h)),
                    abs(corrected_kin.vz - next_kin.vz)
                    / (1.0 + abs(corrected_kin.vz)),
                )
                next_kin = corrected_kin
                if error <= corrector_tolerance:
                    break
            else:
                raise RuntimeError(
                    f"Flight corrector failed to converge at t={next_kin.t:.6g} s"
                )

            # Re-evaluate the complete corrected endpoint for history.
            _, _, end_airspeed, _, _ = self.flight_kinematics(next_kin)
            end_atmosphere = self.env.atmosphere(next_kin.h, end_airspeed)

            end_aero = self.trial_aero(next_kin, end_atmosphere, end_engine_on)
            end_plant = PlantOut(
                aero=end_aero,
                thermal=thermal_out,
                fluids=fluid_state,
            )
            end_forces = self.forces(next_kin, end_plant, mass)
            self._record_twr(result, next_kin, end_forces["twr"], kin, start_forces["twr"])
            self._record_extrema(result, end_atmosphere, end_aero)
            result.max_altitude = max(result.max_altitude, next_kin.h)
            if kin.vz > 0.0 and next_kin.vz <= 0.0:
                fraction = kin.vz / (kin.vz - next_kin.vz)
                elapsed = fraction * kin.dt
                result.apogee = kin.h + 0.5 * kin.vz * elapsed
                result.apogee_time = kin.t + elapsed
                result.max_altitude = max(result.max_altitude, result.apogee)
            if engine_on:
                # Flight-level shutdown timing remains endpoint-sampled until
                # the deferred trajectory event splitting is implemented.
                result.burn_duration += kin.dt
                if not end_engine_on:
                    result.burn_complete = True
            result.final_time = next_kin.t
            result.final_altitude = next_kin.h
            result.final_velocity = next_kin.vz
            result.final_mass = mass
            result.shutdown_reason = fluid_state.propulsion.shutdown_reason
            merge_margins(result.constraints, fluid_state.constraints,
                          times=result.constraint_times, sample_times=fluid_state.constraint_times,
                          time=next_kin.t)
            loads = None
            if compute_loads:
                loads = self._evaluate_loads(result, next_kin, end_atmosphere, end_aero,
                                             end_forces, end_engine_on)
            if record_history:
                state = {
                    "kinematics": next_kin,
                    "atmosphere": end_atmosphere,
                    "plant": end_plant,
                    "forces": end_forces,
                    "mass_properties": self.mass_properties(),
                    "engine_on": end_engine_on,
                    "on_rail": self.on_rail(next_kin.h),
                }
                if loads is not None:
                    state["loads"] = loads
                history.append(state)
            kin = next_kin
            if progress is not None:
                progress(kin)

        result.termination = "apogee" if result.apogee_reached else ("no_ascent" if kin.vz < 0 else "time_limit")
        return history

    def _evaluate_loads(self, result, kin, atmosphere, aero, forces, engine_on):
        loads = self.loads.evaluate(atmosphere.q, atmosphere.Ma, kin.alpha,
                                    aero.A, forces["thrust"], engine_on=engine_on)
        for name in ("axial", "normal", "shear", "bending"):
            values = np.asarray(loads[name])
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Nonfinite {name} loads at t={kin.t}")
            result.load_peaks[name] = max(result.load_peaks.get(name, 0.0), float(np.max(np.abs(values))))
        return loads

    def _record_twr(self, result, kin, twr, previous=None, previous_twr=None):
        result.max_twr = max(result.max_twr, twr)
        if result.rail_exit_time is not None:
            return
        sample_time = kin.t
        if not self.on_rail(kin.h):
            if previous is None or not self.on_rail(previous.h):
                result.rail_exit_time = kin.t
                return
            rail_end = float(self.cfg["launch"]["altitude"]) + float(self.cfg["launch"]["rail_height"])
            fraction = (rail_end - previous.h) / (kin.h - previous.h)
            sample_time = previous.t + fraction * (kin.t - previous.t)
            twr = previous_twr + fraction * (twr - previous_twr)
            result.rail_exit_time = sample_time
        if result.min_rail_twr is None or twr < result.min_rail_twr:
            result.min_rail_twr = twr
            result.min_rail_twr_time = sample_time

    def _record_extrema(self, result, atmosphere, aero):
        result.max_q = max(result.max_q, float(atmosphere.q))
        diameter = self.cfg.get("vehicle", {}).get("OMLD")
        if diameter is not None and math.isfinite(aero.cp):
            stability = (aero.cp - self.vehicle.cg) / float(diameter)
            previous = result.min_stability_calibers
            result.min_stability_calibers = stability if previous is None else min(previous, stability)

    def on_rail(self, altitude: float) -> bool:
        launch = self.cfg["launch"]
        rail_end = float(launch["altitude"]) + float(launch["rail_height"])
        return altitude < rail_end

    def angle_of_attack(self, time: float, altitude: float) -> float:
        """Return zero on the rail and the scheduled AoA after rail exit."""

        return 0.0 if self.on_rail(altitude) else self.aero.aoa(time)

    @staticmethod
    def engine_on(propulsion: PropulsionOut) -> bool:
        """Return whether combustion is active for aerodynamic deck selection."""

        return propulsion.mode == "combusting"
