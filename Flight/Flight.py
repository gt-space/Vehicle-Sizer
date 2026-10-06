from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np

from Fluids.PropSystem import PropSystem
from errors import TrialDomainError
from .flight_forces import gravity
from .loads import Loads
from simulation_types import SimResult
from constraints import OperatingInfeasible, merge_margins
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
    """Coordinate atmosphere, propulsion, vehicle mass, and planar 3DOF kinematics."""

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
        self.thrust_tilt = math.radians(float(cfg.get("engine", {}).get("thrust_tilt_deg", 0.0)))
        if not math.isfinite(self.thrust_tilt):
            raise ValueError("engine.thrust_tilt_deg must be finite")
        self.aero_q_cutoff = float(cfg.get("aero", {}).get("q_cutoff_Pa", 0.0))
        if not math.isfinite(self.aero_q_cutoff) or self.aero_q_cutoff < 0:
            raise ValueError("aero.q_cutoff_Pa must be finite and nonnegative")

    def trial_aero(
        self,
        kin: KinematicsState,
        atm: AtmosState,
        engine_on: bool,
    ) -> AeroOut:
        self.check_trial(kin, atm)
        if atm.q < self.aero_q_cutoff:
            return AeroOut(Cd=float("nan"), D=0.0, Ca=float("nan"), A=0.0,
                           Cn=float("nan"), N=0.0, cp=float("nan"), ballistic_coast=True)
        angle_limit = min(15., float(getattr(self.aero, "alpha_deg", [15.])[-1]))
        if abs(math.degrees(kin.alpha)) > angle_limit:
            raise TrialDomainError(f"Flight trial AoA exceeds aero domain ({angle_limit:g} deg)")
        if atm.Ma > float(getattr(self.aero, "mach", [float("inf")])[-1]):
            raise TrialDomainError("Flight trial Mach exceeds aero domain")
        return self.aero.evaluate(kin, atm, engine_on)

    def check_trial(self, kin, atmosphere=None):
        """Guard model inputs; invalid numerical guesses are recoverable."""
        if (not all(math.isfinite(getattr(kin, key)) for key in
                    ("t", "dt", "x", "h", "vx", "vz", "theta", "q", "alpha", "m", "Iyy"))
                or kin.t < 0 or kin.dt <= 0 or kin.m <= 0 or kin.Iyy <= 0):
            raise TrialDomainError("Flight trial requires finite state, positive dt, mass and inertia")
        if atmosphere is not None:
            values = (atmosphere.T, atmosphere.p, atmosphere.rho, atmosphere.mu,
                      atmosphere.a, atmosphere.q, atmosphere.Ma)
            if (not all(math.isfinite(value) for value in values)
                    or min(atmosphere.T, atmosphere.a, atmosphere.mu) <= 0
                    or min(atmosphere.p, atmosphere.rho, atmosphere.q, atmosphere.Ma) < 0):
                raise TrialDomainError("Flight trial has an invalid atmosphere state")

    def check_commit(self, start, kin):
        """Check a converged endpoint before committing history or thermal state.

        A descending endpoint brackets apogee; the caller must localize it
        before evaluating reversed-flow aerodynamics or accepting the segment.
        """
        crossing = self._apogee_crossing(start, kin)
        if crossing is not None:
            return crossing
        if kin.vz < 0:
            return {"event": "no_ascent", "time_s": kin.t}
        limit = self.cfg.get("constraints", {}).get("max_aoa_deg")
        if limit is not None:
            margin = float(limit) - abs(math.degrees(kin.alpha))
            if margin < 0:
                raise OperatingInfeasible({"max_aoa_deg": margin}, time=kin.t)
        self.check_trial(kin, self._atmosphere(kin))
        return None

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
        *, commit_thermal: bool = True,
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
        if commit_thermal:
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
        start_forces: Dict[str, float],
        end_forces: Dict[str, float],
        mass: float,
        Iyy: float,
    ) -> KinematicsState:
        """Apply the implicit trapezoidal corrector to the predicted endpoint."""

        dt = kin.dt

        # new corrector equations for 3DOF
        vx = kin.vx + 0.5 * (start_forces["ax"] + end_forces["ax"]) * dt

        vz = kin.vz + 0.5 * (start_forces["az"] + end_forces["az"]) * dt

        x = kin.x + 0.5 * (kin.vx + vx) * dt
        h = kin.h + 0.5 * (kin.vz + vz) * dt

        q = kin.q + 0.5 * (start_forces["pitch_acceleration"] + end_forces["pitch_acceleration"]) * dt

        theta = kin.theta + 0.5 * (kin.q + q) * dt

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
        drag = float(plant.aero.D)
        axial_aero = float(plant.aero.A)
        normal_aero = float(plant.aero.N)
        cp = float(plant.aero.cp)
        cg = float(self.vehicle.cg)
        weight = gravity(mass, kin.h)

        if (not all(math.isfinite(x) for x in (
                    thrust,
                    drag,
                    axial_aero,
                    normal_aero,
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
        if not plant.aero.ballistic_coast and not math.isfinite(cp):
            raise ValueError("Nonphysical or nonfinite flight force state")

        # Positive tilt rotates thrust counterclockwise from the body axis.
        thrust_axial = thrust * math.cos(self.thrust_tilt)
        thrust_normal = thrust * math.sin(self.thrust_tilt)
        body_axial = thrust_axial - axial_aero
        body_normal = thrust_normal + normal_aero
        Fx = body_axial * math.cos(kin.theta) - body_normal * math.sin(kin.theta)
        Fz = body_axial * math.sin(kin.theta) + body_normal * math.cos(kin.theta) - weight

        ax = Fx / mass
        az = Fz / mass

        axial_specific_force = body_axial / mass # needed for fluid head pressure

        # Thrust line passes through the existing engine/airframe interface.
        thrust_moment = (-(self.vehicle.engine_start_station - cg) * thrust_normal
                         if thrust_normal else 0.0)
        aero_moment = 0.0 if plant.aero.ballistic_coast else -normal_aero * (cp - cg)
        pitch_moment = aero_moment + thrust_moment
        pitch_acceleration = pitch_moment / kin.Iyy

        return {
            "thrust": thrust,
            "thrust_axial": thrust_axial,
            "thrust_normal": thrust_normal,
            "thrust_moment": thrust_moment,
            "drag": drag,
            "axial_aero": axial_aero,
            "normal_aero": normal_aero,
            "gravity": weight,
            "Fx": Fx,
            "Fz": Fz,
            "ax": ax,
            "az": az,
            "axial_specific_force": axial_specific_force,
            "pitch_moment": pitch_moment,
            "pitch_acceleration": pitch_acceleration,
            "twr": thrust / weight,
            "net": Fz,
            "acceleration": az,
        }

    def mass_properties(self) -> Dict[str, Any]:
        """Snapshot the vehicle mass state so later updates cannot alter history."""

        snapshot = {
            "total_mass": float(self.vehicle.total_mass),
            "cg": float(self.vehicle.cg),
            "Ixx": float(self.vehicle.Ixx),
            "Iyy": float(self.vehicle.Iyy),
            "length": float(self.vehicle.length),
            "diameter": float(self.cfg.get("vehicle", {}).get("OMLD", np.nan)),
            "station": self.vehicle.station.copy(),
            "axial_mass": self.vehicle.mass.copy(),
        }
        for name in ("cell_edges", "cell_widths", "dry_mass"):
            if hasattr(self.vehicle, name):
                snapshot[name] = getattr(self.vehicle, name).copy()
        return snapshot

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
    def constrain_to_rail(kin: KinematicsState) -> KinematicsState:
        return replace(
            kin,
            x=0.0,
            vx=0.0,
            theta=np.pi / 2,
            q=0.0,
            alpha=0.0,
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
            angle = kin.theta - gamma_air
            alpha = math.atan2(math.sin(angle), math.cos(angle)) # angle of attack (wind-relative)
        else:
            gamma_air = kin.theta # flight-path relative to wind
            alpha = 0.0 # on rail

        return speed, gamma, airspeed, gamma_air, alpha


    def _flight_kinematics(self, kin):
        """Evaluate inertial and air-relative kinematics at the trial altitude."""
        wind_x, wind_z = self._wind(kin.h)
        return self.flight_kinematics(kin, wind_x=wind_x, wind_z=wind_z)

    def _wind(self, altitude):
        """Allow atmosphere-only environment providers to represent still air."""
        lookup = getattr(self.env, "wind", None)
        return lookup(altitude) if lookup is not None else (0.0, 0.0)

    def _kinematic_boundary(self, kin, on_rail):
        """Use the segment's fixed rail mode and derive AoA from its trial state."""
        if on_rail:
            return self.constrain_to_rail(kin)
        return replace(kin, alpha=self._flight_kinematics(kin)[4])

    def _atmosphere(self, kin):
        self.check_trial(kin)
        return self.env.atmosphere(kin.h, self._flight_kinematics(kin)[2])

    @staticmethod
    def _apogee_crossing(start, trial):
        """Locate a trial velocity crossing before querying descent-only aerodynamics."""
        if start.vz > 0.0 and trial.vz <= 0.0:
            return {"event": "apogee", "time_s": start.t + start.dt *
                    start.vz / (start.vz - trial.vz)}
        return None

    def _attempt_segment(self, kin, fluid_state, on_rail, event_tolerance):
        """Tentatively advance one fixed-mode segment, without publishing results."""
        settings = self.cfg.get("advanced", {}).get("flight", self.cfg["simulation"])
        corrector_tolerance = float(settings.get("corrector_tolerance", 1e-8))
        corrector_max_iterations = int(settings.get("corrector_max_iterations", 10))
        fluid_solve_post_shutdown = self.cfg["simulation"].get("fluid_solve_post_shutdown", True)
        # Evaluate every force used for propagation at the beginning of the
        # interval, using the fluid and vehicle state at the same time.
        thermal_out = None
        mass = float(self.vehicle.total_mass)
        inertia = float(self.vehicle.Iyy)
        kin = replace(
            kin,
            dt=kin.dt,
            m=mass,
            Iyy=inertia,
        )
        kin = self._kinematic_boundary(kin, on_rail)
        atmosphere = self._atmosphere(kin)
        engine_on = self.engine_on(fluid_state.propulsion)
        aero_out = self.trial_aero(kin, atmosphere, engine_on)
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
        predicted_kin = self._kinematic_boundary(predicted_kin, on_rail)
        crossing = self._apogee_crossing(kin, predicted_kin)
        if crossing is not None:
            return None, fluid_state, None, start_forces, crossing
        predicted_atmosphere = self._atmosphere(predicted_kin)
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
                    axial_specific_force=start_forces["axial_specific_force"],
                    commit_thermal=False,
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
        shutdown = next((event for event in fluid_state.events
                         if event.get("event") == "shutdown" and "before" in event), None)
        if shutdown is not None and shutdown["time_s"] < predicted_kin.t - event_tolerance:
            return None, fluid_state, thermal_out, start_forces, shutdown
        # Integrate the powered segment with its left-hand thrust at cutoff.
        # The accepted output still carries the post-event mode and inventories.
        endpoint_fluid = shutdown["before"] if shutdown is not None else fluid_state
        self.vehicle.update_mass_distribution(endpoint_fluid.node)
        mass = float(self.vehicle.total_mass)
        inertia = float(self.vehicle.Iyy)

        # Iterate the implicit trapezoidal corrector. Propulsion and mass
        # are fixed at their solved endpoint values; atmosphere and drag
        # are updated with each corrected kinematic state.
        end_engine_on = self.engine_on(endpoint_fluid.propulsion)
        next_kin = replace(
            predicted_kin,
            m=mass,
            Iyy=inertia,
        )
        for _ in range(corrector_max_iterations):
            trial_atmosphere = self._atmosphere(next_kin)
            trial_aero = self.trial_aero(
                next_kin,
                trial_atmosphere,
                end_engine_on,
            )
            trial_plant = PlantOut(
                aero=trial_aero,
                thermal=None,
                fluids=endpoint_fluid,
            )
            end_forces = self.forces(
                next_kin,
                trial_plant,
                mass,
            )
            corrected_kin = self.correct_kinematics(
                kin,
                start_forces,
                end_forces,
                mass,
                inertia,
            )
            corrected_kin = self._kinematic_boundary(corrected_kin, on_rail)
            crossing = self._apogee_crossing(kin, corrected_kin)
            if crossing is not None:
                return None, fluid_state, thermal_out, start_forces, crossing
            error = max(
                abs(getattr(corrected_kin, name) - getattr(next_kin, name))
                / (1.0 + abs(getattr(corrected_kin, name)))
                for name in ("x", "h", "vx", "vz", "theta", "q")
            )
            next_kin = corrected_kin
            if error <= corrector_tolerance:
                break
        else:
            raise TrialDomainError(
                f"Flight corrector failed to converge at t={next_kin.t:.6g} s"
            )

        stopping = self.check_commit(kin, next_kin)
        if stopping is not None:
            return None, fluid_state, thermal_out, start_forces, stopping
        # Validate the final correction inside the retry boundary as well.
        self.trial_aero(next_kin, self._atmosphere(next_kin), end_engine_on)
        return next_kin, fluid_state, thermal_out, start_forces, shutdown

    def run(self, h0: float = 0.0, v0: float = 0.0, *, progress=None,
            record_history: bool = True, compute_loads: bool = True) -> List[Dict[str, Any]]:
        """Run the planar 3DOF trajectory with an explicit-implicit predictor-corrector."""

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
        event_tolerance = float(advanced.get("event_time_tolerance", 1e-6))
        event_max_iterations = advanced.get("event_max_iterations", 50)
        trial_max_retries = advanced.get("trial_max_retries", 10)
        trial_min_dt = float(advanced.get("trial_min_dt", 1e-6))
        if (isinstance(trial_max_retries, bool) or not isinstance(trial_max_retries, int)
                or trial_max_retries < 0 or not math.isfinite(trial_min_dt) or trial_min_dt <= 0):
            raise ValueError("Flight requires integer trial_max_retries >= 0 and finite trial_min_dt > 0")
        if not math.isfinite(event_tolerance) or event_tolerance <= 0:
            raise ValueError("Flight event_time_tolerance must be positive and finite")
        if (isinstance(event_max_iterations, bool) or not isinstance(event_max_iterations, int)
                or event_max_iterations < 1):
            raise ValueError("Flight event_max_iterations must be a positive integer")

        inertia = float(self.vehicle.Iyy)
        rail_mode = self.on_rail(h0)
        kin = KinematicsState(
            t=0.0,
            dt=dt,
            x=0.0,
            h=h0,
            vx=0.0,
            vz=v0,
            theta=math.pi / 2,
            q=0.0,
            alpha=0.0,
            m=mass,
            Iyy=inertia,
        )
        kin = self._kinematic_boundary(kin, rail_mode)
        atmosphere = self._atmosphere(kin)

        fluid_state = self.commit_fluid(
            dt=None,
            atm=atmosphere,
            axial_specific_force=gravity(mass, h0) / mass,
        )
        self.vehicle.update_mass_distribution(fluid_state.node)
        mass = float(self.vehicle.total_mass)
        inertia = float(self.vehicle.Iyy)
        kin = replace(kin, m=mass, Iyy=inertia)
        history: List[Dict[str, Any]] = []
        result = self.result = SimResult(
            max_altitude=h0, initial_mass=mass, final_mass=mass,
            dry_mass=float(np.sum(getattr(self.vehicle, "dry_mass", self.vehicle.mass))),
            final_altitude=h0, final_velocity=v0,
            final_vz=v0, final_speed=abs(v0), final_pitch_angle=math.pi / 2,
            geometry_constraints=dict(getattr(self.vehicle, "geometry_constraints", {})),
            history=history if record_history else None,
            constraints=dict(fluid_state.constraints),
            constraint_times=dict(fluid_state.constraint_times),
        )
        thermal_out = None
        result.burn_complete = fluid_state.propulsion.mode == "shutdown"
        result.shutdown_reason = fluid_state.propulsion.shutdown_reason
        if self.check_commit(kin, kin) is not None:
            result.termination = "no_ascent"
            return history
        initial_aero = self.trial_aero(kin, atmosphere, self.engine_on(fluid_state.propulsion))
        initial_forces = self.forces(kin, PlantOut(initial_aero, None, fluid_state), mass)
        if record_history:
            result.initial_state = {
                "kinematics": kin,
                "atmosphere": atmosphere,
                "plant": PlantOut(initial_aero, None, fluid_state),
                "forces": initial_forces,
                "mass_properties": self.mass_properties(),
                "engine_on": self.engine_on(fluid_state.propulsion),
            }
        self._record_twr(result, kin, initial_forces["twr"])
        if compute_loads:
            self._evaluate_loads(result, kin, atmosphere, initial_aero, initial_forces,
                                 self.engine_on(fluid_state.propulsion))
        if progress is not None:
            progress(kin)

        rail_end = float(self.cfg["launch"]["altitude"]) + float(self.cfg["launch"]["rail_height"])
        macro_index = 0
        while kin.t < t_end and not result.apogee_reached and (kin.t == 0.0 or kin.vz >= 0.0):
            macro_end = min((macro_index + 1) * dt, t_end)
            start_kin, start_fluid = kin, fluid_state
            start_atmosphere = self._atmosphere(kin)
            start_sample = self._kinematic_boundary(kin, rail_mode)
            self._record_extrema(result, start_atmosphere, self.trial_aero(
                start_sample, start_atmosphere, self.engine_on(start_fluid.propulsion)), start_sample)
            checkpoint = (self.prop_system.checkpoint() if fluid_solve_post_shutdown
                          or start_fluid.propulsion.mode != "shutdown" else None)
            duration = macro_end - kin.t
            low, low_height, high, high_height = 0.0, kin.h, None, None
            rail_crossing = False
            trial_retries = 0
            try:
                for attempt in range(event_max_iterations):
                    if attempt:
                        if checkpoint is not None:
                            self.prop_system.restore(checkpoint)
                        self.vehicle.update_mass_distribution(start_fluid.node)
                    kin = self._kinematic_boundary(replace(start_kin, dt=duration), rail_mode)
                    try:
                        next_kin, fluid_state, thermal_out, start_forces, shutdown = self._attempt_segment(
                            kin, start_fluid, rail_mode, event_tolerance)
                    except TrialDomainError as error:
                        if trial_retries >= trial_max_retries or duration / 2 < trial_min_dt:
                            raise RuntimeError(f"Flight trial recovery exhausted at t={start_kin.t:g} s "
                                               f"after {trial_retries} reductions: {error}") from error
                        trial_retries += 1
                        duration /= 2
                        low, low_height, high, high_height = 0.0, start_kin.h, None, None
                        continue
                    if next_kin is None:
                        if shutdown["event"] == "no_ascent":
                            if checkpoint is not None:
                                self.prop_system.restore(checkpoint)
                            self.vehicle.update_mass_distribution(start_fluid.node)
                            result.termination = "no_ascent"
                            return history
                        # Re-solve from the checkpoint with the shorter boundary
                        # prediction. Approach apogee from ascent, shutdown from
                        # the right so the IDAS mode change is included.
                        offset = -.25 if shutdown["event"] == "apogee" else .25
                        duration = min(duration, max(0.0, shutdown["time_s"] - kin.t)
                                       + offset * event_tolerance)
                        if duration <= 0.0:
                            raise RuntimeError("Flight event interval is below time resolution")
                        low, low_height, high, high_height = 0.0, kin.h, None, None
                        continue
                    if rail_mode:
                        if next_kin.h >= rail_end:
                            high, high_height = duration, next_kin.h
                            if (high - low <= event_tolerance or
                                    (next_kin.h - rail_end) / max(abs(next_kin.vz), 1e-12)
                                    <= event_tolerance):
                                rail_crossing = True
                                break
                        elif high is None:
                            break  # Neither event was crossed.
                        else:
                            low, low_height = duration, next_kin.h
                        fraction = (rail_end - low_height) / (high_height - low_height)
                        duration = low + max(.1, min(.9, fraction)) * (high - low)
                        continue
                    break
                else:
                    raise RuntimeError(f"Flight event localization failed at t={start_kin.t:g} "
                                       f"after {event_max_iterations} attempts")
            except Exception:
                if checkpoint is not None:
                    self.prop_system.restore(checkpoint)
                self.vehicle.update_mass_distribution(start_fluid.node)
                raise

            # Only accepted segments may alter thermal state or flight results.
            if thermal_out is not None:
                self.thermal.commit(thermal_out)
            engine_on = self.engine_on(start_fluid.propulsion)
            end_engine_on = self.engine_on(fluid_state.propulsion)
            self.vehicle.update_mass_distribution(fluid_state.node)
            mass = float(self.vehicle.total_mass)
            next_kin = replace(next_kin, m=mass, Iyy=float(self.vehicle.Iyy))
            # Sample the left side of either discontinuity before switching the
            # flight mode; the history record below describes the right side.
            if rail_crossing or shutdown is not None:
                left_fluid = shutdown["before"] if shutdown is not None else fluid_state
                self.vehicle.update_mass_distribution(left_fluid.node)
                left_mass = float(self.vehicle.total_mass)
                left_atmosphere = self._atmosphere(next_kin)
                left_aero = self.trial_aero(next_kin, left_atmosphere,
                                            self.engine_on(left_fluid.propulsion))
                left_forces = self.forces(next_kin, PlantOut(left_aero, thermal_out, left_fluid), left_mass)
                self._record_twr(result, next_kin, left_forces["twr"], kin, start_forces["twr"])
                self._record_extrema(result, left_atmosphere, left_aero, next_kin)
                if compute_loads:
                    self._evaluate_loads(result, next_kin, left_atmosphere, left_aero,
                                         left_forces, self.engine_on(left_fluid.propulsion))
                self.vehicle.update_mass_distribution(fluid_state.node)
            if rail_crossing:
                rail_mode = False
                result.rail_exit_time = next_kin.t
                next_kin = self._kinematic_boundary(next_kin, False)

            # Re-evaluate the complete corrected endpoint for history.
            end_atmosphere = self._atmosphere(next_kin)
            end_aero = self.trial_aero(next_kin, end_atmosphere, end_engine_on)
            end_plant = PlantOut(
                aero=end_aero,
                thermal=thermal_out,
                fluids=fluid_state,
            )
            end_forces = self.forces(next_kin, end_plant, mass)
            self._record_twr(result, next_kin, end_forces["twr"], kin, start_forces["twr"])
            self._record_extrema(result, end_atmosphere, end_aero, next_kin)
            result.max_altitude = max(result.max_altitude, next_kin.h)
            if (kin.vz > 0.0 and end_forces["az"] < 0.0
                    and 0.0 <= next_kin.vz <= -end_forces["az"] * event_tolerance):
                # The remaining ascent lasts at most event_time_tolerance;
                # its height correction is second order in that tolerance.
                remaining = next_kin.vz / -end_forces["az"]
                result.apogee = next_kin.h + .5 * next_kin.vz * remaining
                result.apogee_time = next_kin.t
                result.max_altitude = max(result.max_altitude, result.apogee)
            if engine_on:
                powered_dt = kin.dt if shutdown is None else max(0.0, min(kin.dt, shutdown["time_s"] - kin.t))
                result.burn_duration += powered_dt
                if not end_engine_on:
                    result.burn_complete = True
            result.final_time = next_kin.t
            result.final_altitude = next_kin.h
            result.final_velocity = next_kin.vz  # compatibility: vertical velocity
            result.final_x = next_kin.x
            result.final_vx = next_kin.vx
            result.final_vz = next_kin.vz
            result.final_speed = math.hypot(next_kin.vx, next_kin.vz)
            result.final_pitch_angle = next_kin.theta
            result.final_pitch_rate = next_kin.q
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
                wind_x, wind_z = self._wind(next_kin.h)
                _, _, airspeed, gamma_air, _ = self.flight_kinematics(
                    next_kin,
                    wind_x=wind_x,
                    wind_z=wind_z,
                )
                state = {
                    "kinematics": next_kin,
                    "atmosphere": end_atmosphere,
                    "plant": end_plant,
                    "forces": end_forces,
                    "mass_properties": self.mass_properties(),
                    "wind": {
                        "wind_x": float(wind_x),
                        "wind_z": float(wind_z),
                        "airspeed": float(airspeed),
                        "gamma_air": float(gamma_air),
                    },
                    "engine_on": end_engine_on,
                    "on_rail": rail_mode,
                }
                if loads is not None:
                    state["loads"] = loads
                history.append(state)
            kin = next_kin
            if kin.t >= macro_end:
                macro_index += 1
            if progress is not None:
                progress(kin)

        result.termination = "apogee" if result.apogee_reached else ("no_ascent" if kin.vz < 0 else "time_limit")
        return history

    def _evaluate_loads(self, result, kin, atmosphere, aero, forces, engine_on):
        loads = self.loads.evaluate(atmosphere.q, atmosphere.Ma, kin.alpha,
                                    aero.A, forces["thrust_axial"], engine_on=engine_on,
                                    thrust_normal=forces["thrust_normal"],
                                    aerodynamic=not aero.ballistic_coast)
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

    def _record_extrema(self, result, atmosphere, aero, kin):
        result.max_aoa_deg = max(result.max_aoa_deg, abs(math.degrees(kin.alpha)))
        limit = self.cfg.get("constraints", {}).get("max_aoa_deg")
        if limit is not None:
            merge_margins(result.constraints, {"max_aoa_deg": float(limit) - result.max_aoa_deg},
                          times=result.constraint_times, time=kin.t)
        result.max_q = max(result.max_q, float(atmosphere.q))
        diameter = self.cfg.get("vehicle", {}).get("OMLD")
        if diameter is not None and math.isfinite(aero.cp):
            stability = (aero.cp - self.vehicle.cg) / float(diameter)
            previous = result.min_stability_calibers
            result.min_stability_calibers = stability if previous is None else min(previous, stability)
            fraction = (aero.cp - self.vehicle.cg) / self.vehicle.length
            previous = result.min_stability_length_fraction
            result.min_stability_length_fraction = fraction if previous is None else min(previous, fraction)
            previous = result.max_stability_length_fraction
            result.max_stability_length_fraction = fraction if previous is None else max(previous, fraction)

    def on_rail(self, altitude: float) -> bool:
        launch = self.cfg["launch"]
        rail_end = float(launch["altitude"]) + float(launch["rail_height"])
        return altitude < rail_end

    def angle_of_attack(self, time: float, altitude: float) -> float:
        """Legacy schedule helper; propagation derives AoA from the 3DOF state."""

        return 0.0 if self.on_rail(altitude) else self.aero.aoa(time)

    @staticmethod
    def engine_on(propulsion: PropulsionOut) -> bool:
        """Return whether combustion is active for aerodynamic deck selection."""

        return propulsion.mode == "combusting"
