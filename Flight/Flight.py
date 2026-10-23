from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Optional

import numpy as np

from Fluids.PropSystem import PropSystem
from diagnostics.errors import TrialDomainError
from .flight_forces import gravity
from .loads import Loads
from .integration import advance_flight, sample_interval
from .canards import CanardSystem
from .roll import RollAnalysis
from simulation_types import SimResult
from diagnostics.constraints import merge_records
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
        self.rail_angle = math.radians(float(cfg.get('launch', {}).get('rail_inclination_deg', 90.0)))
        if not math.isfinite(self.rail_angle) or not 0 < self.rail_angle < math.pi:
            raise ValueError('launch.rail_inclination_deg must be finite and between 0 and 180 degrees')
        self.rail_sin = math.sin(self.rail_angle)
        self.rail_cos = 0.0 if self.rail_angle == math.pi / 2 else math.cos(self.rail_angle)
        self.env = env
        self.aero = aero
        self.prop_system = prop_system
        self.vehicle = vehicle
        self.thermal = thermal
        self.loads = Loads(vehicle, aero)
        self.canards = (CanardSystem(cfg['canards'], cfg['vehicle']['OMLD'], vehicle)
                        if cfg.get('canards') is not None else None)
        self.roll_analysis = (RollAnalysis(cfg['vehicle']['OMLD'], cfg.get('roll', {}), aero)
                              if cfg.get('roll') is not None or self.canards is not None else None)
        self.thrust_tilt = math.radians(float(cfg.get("engine", {}).get("thrust_tilt_deg", 0.0)))
        if not math.isfinite(self.thrust_tilt):
            raise ValueError("engine.thrust_tilt_deg must be finite")
        self.aero_q_cutoff = float(cfg.get("aero", {}).get("q_cutoff_Pa", 0.0))
        if not math.isfinite(self.aero_q_cutoff) or self.aero_q_cutoff < 0:
            raise ValueError("aero.q_cutoff_Pa must be finite and nonnegative")

        settings = cfg.get("advanced", {}).get("heating", {})
        self.heat_tolerance = float(settings.get("convergence_tolerance", 1.0e-3))
        self.temperature_tolerance = float(settings.get("temperature_tolerance", 1.0e-2))
        self.heat_iterations = int(settings.get("max_iterations", 5))
        if (not all(math.isfinite(v) for v in (self.heat_tolerance, self.temperature_tolerance))
                or min(self.heat_tolerance, self.temperature_tolerance) < 0.0):
            raise ValueError("Heating convergence tolerances must be nonnegative")
        if self.heat_iterations < 1:
            raise ValueError("Heating convergence iterations must be positive")

    def trial_aero(self, atm: AtmosState, alpha: float, engine_on: bool) -> AeroOut:
        self.check_trial(atmosphere=atm)
        if not math.isfinite(alpha):
            raise TrialDomainError('Flight trial requires finite AoA')
        if atm.q < self.aero_q_cutoff:
            return AeroOut(Cd=float("nan"), D=0., Ca=float("nan"), A=0.,
                           Cn=float("nan"), N=0., cp=float("nan"), ballistic_coast=True)
        angle_limit = min(15., float(getattr(self.aero, "alpha_deg", [15.])[-1]))
        if abs(math.degrees(alpha)) > angle_limit:
            raise TrialDomainError(f"Flight trial AoA exceeds aero domain ({angle_limit:g} deg)")
        if atm.Ma > float(getattr(self.aero, "mach", [float("inf")])[-1]):
            raise TrialDomainError("Flight trial Mach exceeds aero domain")
        return self.aero.evaluate(atm, alpha, engine_on)

    @staticmethod
    def check_trial(kin=None, atmosphere=None):
        if kin is not None and (not all(math.isfinite(value) for value in vars(kin).values()) or kin.t < 0):
            raise TrialDomainError("Flight trial requires finite state and nonnegative time")
        if atmosphere is not None:
            values = vars(atmosphere).values()
            if (not all(math.isfinite(value) for value in values)
                    or min(atmosphere.T, atmosphere.a, atmosphere.mu) <= 0
                    or min(atmosphere.p, atmosphere.rho, atmosphere.q, atmosphere.Ma) < 0):
                raise TrialDomainError("Flight trial has an invalid atmosphere state")

    def update_fluid(
        self,
        dt: Optional[float],
        atm: AtmosState,
        thermal_out: Optional[ThermalOut] = None,
        axial_specific_force: float = 0.0,
        *,
        commit: bool,
    ) -> FluidOut:
        """Pass thermal boundaries to a preview or committed fluid update."""

        return self.prop_system.update(
            dt=dt,
            atm=atm,
            heat_rate=thermal_out.heat_rates() if thermal_out is not None else {},
            commit=commit,
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
        dt: float,
        alpha: float,
        atm: AtmosState,
        fluids: FluidOut,
        axial_specific_force: float,
        *, commit_thermal: bool = True,
    ) -> tuple[FluidOut, Optional[ThermalOut]]:
        """Converge external heat boundaries, then commit the fluid step once."""

        if self.thermal is None:
            return (
                self.update_fluid(
                    dt,
                    atm,
                    commit=True,
                    axial_specific_force=axial_specific_force,
                ),
                None,
            )

        thermal_out = self.thermal.trial(
            dt,
            atm,
            fluids,
            alpha=alpha,
            axial_specific_force=axial_specific_force,
        )
        for _ in range(self.heat_iterations):
            trial_fluids = self.update_fluid(
                dt,
                atm,
                commit=False,
                thermal_out=thermal_out,
                axial_specific_force=axial_specific_force,
            )
            next_thermal = self.thermal.trial(
                dt,
                atm,
                trial_fluids,
                alpha=alpha,
                previous=thermal_out,
                axial_specific_force=axial_specific_force,
            )
            if (
                self.heat_rate_error(thermal_out, next_thermal) <= self.heat_tolerance
                and self.wall_temperature_error(thermal_out, next_thermal)
                <= self.temperature_tolerance
            ):
                thermal_out = next_thermal
                break
            thermal_out = next_thermal
        else:
            raise RuntimeError("Fluid/thermal coupling failed to converge")

        fluids = self.update_fluid(
            dt,
            atm,
            commit=True,
            thermal_out=thermal_out,
            axial_specific_force=axial_specific_force,
        )
        if commit_thermal:
            self.thermal.commit(thermal_out)
        return fluids, thermal_out

    @staticmethod
    def predict_kinematics(kin: KinematicsState, forces: Dict[str, float], dt: float) -> KinematicsState:
        """Explicitly predict the endpoint using already evaluated start forces."""

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
            x=x,
            h=h,
            vx=vx,
            vz=vz,
            theta=theta,
            q=q,
        )

    def forces(
        self,
        kin: KinematicsState,
        aero: AeroOut,
        propulsion: PropulsionOut,
        mass_properties: dict,
    ) -> Dict[str, float]:
        """
        Forces, moments, and state derivatives needed to update the 3DOF state vector.
        From this, only the accelerations and pitch rate outputs really matter.
        """

        mass, cg, inertia = (mass_properties[key] for key in ("total_mass", "cg", "Iyy"))
        thrust = float(propulsion.thrust)
        drag = float(aero.D)
        axial_aero = float(aero.A)
        normal_aero = float(aero.N)
        cp = float(aero.cp)
        canard = aero.canard
        if canard is not None:
            drag += canard.drag_n
            axial_aero += canard.axial_force_n
            normal_aero += canard.normal_force_n
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
                    inertia,
                )
            ) or mass <= 0.0 or weight <= 0.0 or inertia <= 0.0
        ):
            raise ValueError("Nonphysical or nonfinite flight force state")
        if not aero.ballistic_coast and not math.isfinite(cp):
            raise ValueError("Nonphysical or nonfinite flight force state")

        # Positive tilt rotates counterclockwise from the body axis
        thrust_axial = thrust * math.cos(self.thrust_tilt)
        thrust_normal = thrust * math.sin(self.thrust_tilt)
        body_axial = thrust_axial - axial_aero
        body_normal = thrust_normal + normal_aero
        Fx = body_axial * math.cos(kin.theta) - body_normal * math.sin(kin.theta)
        Fz = body_axial * math.sin(kin.theta) + body_normal * math.cos(kin.theta) - weight

        ax = Fx / mass
        az = Fz / mass

        axial_specific_force = body_axial / mass # needed for fluid head pressure


        thrust_moment = (-(self.vehicle.engine_start_station - cg) * thrust_normal
                         if thrust_normal else 0.0)
        aero_moment = 0.0 if aero.ballistic_coast else -aero.N * (cp - cg)
        if canard is not None:
            aero_moment += canard.pitch_moment_nm
        pitch_moment = aero_moment + thrust_moment
        pitch_acceleration = pitch_moment / inertia

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
            samples=(), events=(),
            mdot={branch_id: 0.0 for branch_id in fluids.mdot},
            propulsion=replace(
                fluids.propulsion,
                thrust=0.0,
                mdot_ox=0.0,
                mdot_fuel=0.0,
                mdot_nozzle=0.0,
            ),
        )

    def constrain_to_rail(self, kin: KinematicsState) -> KinematicsState:
        origin = float(self.cfg['launch']['altitude'])
        distance = kin.x * self.rail_cos + (kin.h - origin) * self.rail_sin
        speed = kin.vx * self.rail_cos + kin.vz * self.rail_sin
        return replace(kin, x=distance * self.rail_cos, h=origin + distance * self.rail_sin,
                       vx=speed * self.rail_cos, vz=speed * self.rail_sin,
                       theta=self.rail_angle, q=0.0)


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

    def _atmosphere(self, kin):
        self.check_trial(kin)
        atmosphere = self.env.atmosphere(kin.h, self._flight_kinematics(kin)[2])
        self.check_trial(kin, atmosphere)
        return atmosphere

    def evaluate_state(self, kin, propulsion, mass_properties, on_rail=False, side='right'):
        """Pure stage evaluation: derive wind/AoA, atmosphere, aero and forces."""
        atmosphere = self._atmosphere(kin)
        alpha = 0. if on_rail else self._flight_kinematics(kin)[4]
        aero = self.trial_aero(atmosphere, alpha, self.engine_on(propulsion))
        if self.canards is not None and not aero.ballistic_coast:
            active = (not on_rail and (kin.t > self.canards.control_start
                      or (kin.t == self.canards.control_start and side == 'right')))
            forcing = self.roll_analysis.forcing(atmosphere)
            aero.canard = self.canards.evaluate(atmosphere, alpha, forcing, active, mass_properties['cg'])
        return atmosphere, aero, self.forces(kin, aero, propulsion, mass_properties), alpha

    def rhs(self, kin, samples, on_rail, side='right'):
        propulsion, mass = sample_interval(kin.t, samples, side)
        # Extend the ascent model only for stages beyond the terminal vz=0
        # guard. Dense event location discards that descending part of the step.
        stage = replace(kin, vz=max(0., kin.vz))
        _, _, force, _ = self.evaluate_state(stage, propulsion, mass, on_rail, side)
        if on_rail:
            acceleration = force['ax'] * self.rail_cos + force['az'] * self.rail_sin
            return [kin.vx, kin.vz, acceleration * self.rail_cos, acceleration * self.rail_sin, 0., 0.]
        return [kin.vx, kin.vz, force['ax'], force['az'], kin.q, force['pitch_acceleration']]

    def prepare_samples(self, fluids):
        """Evaluate vehicle mass properties once per detached fluid snapshot."""
        samples = []
        for time, fluid in fluids.samples:
            self.vehicle.update_mass_distribution(fluid.node)
            samples.append((time, fluid, self.mass_properties()))
        return samples

    def run(self, h0=0., v0=0., *, progress=None, record_history=True, compute_loads=True):
        """Solve the plant on coupling intervals and flight on adaptive RK steps."""
        simulation = self.cfg['simulation']
        dt, t_end = float(simulation['dt']), float(simulation['t_end'])
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(t_end) or t_end < 0:
            raise ValueError('Simulation requires finite dt > 0 and t_end >= 0')
        post_shutdown = simulation.get('fluid_solve_post_shutdown', True)
        if not isinstance(post_shutdown, bool):
            raise ValueError('simulation.fluid_solve_post_shutdown must be boolean')
        settings = dict(rtol=1e-7, position_atol=1e-5, velocity_atol=1e-6,
                        angle_atol=1e-9, angular_rate_atol=1e-9, max_step=2.,
                        event_time_tolerance=1e-8, event_max_iterations=50,
                        trial_min_dt=1e-7, trial_max_retries=20)
        configured = self.cfg.get('advanced', {}).get('flight', {})
        if configured.keys() - settings.keys():
            raise ValueError(f'Unknown flight settings: {sorted(configured.keys() - settings.keys())}')
        settings.update(configured)
        for name in ('rtol', 'position_atol', 'velocity_atol', 'angle_atol',
                     'angular_rate_atol', 'max_step', 'event_time_tolerance', 'trial_min_dt'):
            if not math.isfinite(settings[name]) or settings[name] <= 0:
                raise ValueError(f'Flight {name} must be positive and finite')
        for name in ('event_max_iterations', 'trial_max_retries'):
            if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < 1:
                raise ValueError(f'Flight {name} must be a positive integer')
        kin = KinematicsState(0., 0., h0, v0 * self.rail_cos, v0 * self.rail_sin, self.rail_angle, 0.)
        if self.roll_analysis is not None:
            self.roll_analysis.reset()
        rail = self.on_rail(h0)
        fluid = self.update_fluid(None, self._atmosphere(kin), commit=True,
                                 axial_specific_force=gravity(1., h0) * self.rail_sin)
        self.vehicle.update_mass_distribution(fluid.node)
        mass = self.mass_properties()
        history = []
        result = self.result = SimResult(max_altitude=h0, initial_mass=mass['total_mass'],
            final_mass=mass['total_mass'], dry_mass=float(np.sum(getattr(self.vehicle, 'dry_mass', self.vehicle.mass))),
            final_altitude=h0, final_velocity=kin.vz, final_vx=kin.vx, final_vz=kin.vz, final_speed=abs(v0),
            final_pitch_angle=kin.theta, burn_complete=fluid.propulsion.mode == 'shutdown',
            shutdown_reason=fluid.propulsion.shutdown_reason,
            history=history if record_history else None, constraint_records=dict(fluid.constraint_records))
        result.roll = self.roll_analysis.result if self.roll_analysis is not None else None
        if v0 < 0:
            result.termination = 'no_ascent'
            return history
        evaluation = self.evaluate_state(kin, fluid.propulsion, mass, rail)
        result.initial_state = self._collect_sample(result, kin, evaluation, fluid, None, mass, rail, compute_loads)
        self._record_extrema(result, evaluation[0], evaluation[1], evaluation[3], mass['cg'])
        self._record_roll(kin, evaluation, mass, rail)
        self._record_twr(result, kin, evaluation[2]['twr'])
        thermal = None
        if progress:
            progress(kin)
        rail_height = self.rail_tip_altitude
        while kin.t < t_end:
            launch_acceleration = (evaluation[2]['ax'] * self.rail_cos + evaluation[2]['az'] * self.rail_sin
                                   if rail else evaluation[2]['az'])
            if kin.vz < 0 or (kin.vz == 0 and launch_acceleration <= 0):
                result.termination = 'no_ascent'
                break
            frozen = not post_shutdown and fluid.propulsion.mode == 'shutdown'
            # With frozen inventories and no walls, all remaining coast can use
            # this one interval. RK still adapts and records accepted states.
            end = t_end if frozen and self.thermal is None else min(kin.t + dt, t_end)
            checkpoint = None if frozen else self.prop_system.checkpoint()
            start, start_fluid = kin, fluid
            force = evaluation[2]
            predicted = self.predict_kinematics(start, force, end - start.t)
            atmosphere = self._atmosphere(predicted) if not frozen else evaluation[0]
            alpha = 0. if rail else self._flight_kinematics(predicted)[4]
            try:
                for attempt in range(settings['event_max_iterations']):
                    duration = end - start.t
                    if frozen:
                        fluid = self.stop_fluids(start_fluid)
                        samples = [(start.t, fluid, mass), (end, fluid, mass)]
                        thermal = (self.thermal.trial(duration, atmosphere, fluid, alpha=alpha,
                                   axial_specific_force=force['axial_specific_force']) if self.thermal else None)
                    else:
                        fluid, thermal = self.step_coupled(duration, alpha, atmosphere, start_fluid,
                            force['axial_specific_force'], commit_thermal=False)
                        samples = self.prepare_samples(fluid)
                    stops = {time for time, _, _ in samples if time > start.t}
                    if self.canards is not None and start.t < self.canards.control_start <= end:
                        stops.add(self.canards.control_start)
                    stops = sorted(stops)
                    accepted = advance_flight(start, stops,
                        lambda state, mode, side: self.rhs(state, samples, mode, side),
                        rail, rail_height, settings)
                    terminal = accepted[-1][0]
                    if frozen and self.thermal is not None and terminal.t < end:
                        thermal = self.thermal.trial(terminal.t - start.t, atmosphere, fluid,
                            alpha=alpha, axial_specific_force=force['axial_specific_force'])
                    if not frozen and terminal.t < end:
                        # Only a terminal flight event invalidates the plant's
                        # future. Keep the same BCs while shortening its solve.
                        self.prop_system.restore(checkpoint)
                        if end - terminal.t <= settings['event_time_tolerance']:
                            fluid, thermal = self.step_coupled(terminal.t - start.t, alpha, atmosphere,
                                start_fluid, force['axial_specific_force'], commit_thermal=False)
                            samples = self.prepare_samples(fluid)
                            break
                        end = terminal.t
                        continue
                    break
                else:
                    raise RuntimeError('Flight terminal synchronization failed')
            except Exception:
                if checkpoint is not None:
                    self.prop_system.restore(checkpoint)
                self.vehicle.update_mass_distribution(start_fluid.node)
                raise
            if thermal is not None:
                self.thermal.commit(thermal)
            previous = start
            for kin, mode, event in accepted:
                propulsion, properties = sample_interval(kin.t, samples, 'left')
                evaluation = self.evaluate_state(kin, propulsion, properties, mode, 'left')
                self._record_extrema(result, evaluation[0], evaluation[1], evaluation[3], properties['cg'])
                self._record_twr(result, kin, evaluation[2]['twr'])
                if self.engine_on(sample_interval(previous.t, samples)[0]):
                    result.burn_duration += kin.t - previous.t
                if event == 'rail_release':
                    result.rail_exit_time = kin.t
                    rail = False
                    evaluation = self.evaluate_state(kin, propulsion, properties, rail)
                    self._record_extrema(result, evaluation[0], evaluation[1], evaluation[3], properties['cg'])
                else:
                    rail = mode
                if self.canards is not None and kin.t == self.canards.control_start:
                    evaluation = self.evaluate_state(kin, propulsion, properties, rail)
                    self._record_extrema(result, evaluation[0], evaluation[1], evaluation[3], properties['cg'])
                # Full reports use solved fluid times; frozen fluids are valid
                # at every coast step. RK stages never become history rows.
                matches = [(out, props) for time, out, props in samples if time == kin.t]
                if matches or frozen or event == 'apogee':
                    left, left_mass = matches[0] if matches else (fluid, properties)
                    right, mass = matches[-1] if matches else (fluid, properties)
                    self.vehicle.update_mass_distribution(right.node)
                    before = None
                    if len(matches) > 1:
                        self.vehicle.update_mass_distribution(left.node)
                        left_evaluation = self.evaluate_state(kin, left.propulsion, left_mass, mode, 'left')
                        self._record_extrema(result, left_evaluation[0], left_evaluation[1],
                                             left_evaluation[3], left_mass['cg'])
                        before = self._collect_sample(result, kin, left_evaluation, replace(left, events=()),
                                                      None, left_mass, mode, compute_loads)
                        self.vehicle.update_mass_distribution(right.node)
                    right = replace(right, samples=(), events=tuple(e for e in fluid.events if e['time_s'] == kin.t))
                    evaluation = self.evaluate_state(kin, right.propulsion, mass, rail)
                    self._record_extrema(result, evaluation[0], evaluation[1], evaluation[3], mass['cg'])
                    self._record_twr(result, kin, evaluation[2]['twr'])
                    state = self._collect_sample(result, kin, evaluation, right,
                        thermal if kin.t == terminal.t else None, mass, rail, compute_loads)
                    if state is not None:
                        if before is not None:
                            state['before_event'] = before
                        history.append(state)
                if self.roll_analysis is not None:
                    propulsion, properties = sample_interval(kin.t, samples)
                    self._record_roll(kin, evaluation, properties, rail)
                if event == 'apogee':
                    result.apogee, result.apogee_time = kin.h, kin.t
                    result.termination = 'apogee'
                previous = kin
            # Restore the vehicle to the final synchronized plant snapshot.
            self.vehicle.update_mass_distribution(fluid.node)
            mass = self.mass_properties()
            evaluation = self.evaluate_state(kin, fluid.propulsion, mass, rail)
            result.max_altitude = max(result.max_altitude, kin.h)
            result.final_time, result.final_altitude = kin.t, kin.h
            result.final_velocity = result.final_vz = kin.vz
            result.final_x, result.final_vx = kin.x, kin.vx
            result.final_speed = math.hypot(kin.vx, kin.vz)
            result.final_pitch_angle, result.final_pitch_rate = kin.theta, kin.q
            result.final_mass = mass['total_mass']
            result.burn_complete = fluid.propulsion.mode == 'shutdown'
            result.shutdown_reason = fluid.propulsion.shutdown_reason
            merge_records(result.constraint_records, fluid.constraint_records)
            if progress:
                progress(kin)
            if result.apogee_reached:
                break
        return history

    def _collect_sample(self, result, kin, evaluation, fluids, thermal, mass, on_rail, compute_loads):
        """Collect an accepted state, including either side of a flight event."""
        atmosphere, aero, forces, alpha = evaluation
        if aero.pitch_stiffness_nm_rad <= 0:
            aero.pitch_hz = math.sqrt(-aero.pitch_stiffness_nm_rad / mass['Iyy']) / (2 * math.pi)
        plant = PlantOut(aero, thermal, replace(fluids, samples=()))
        engine_on = self.engine_on(fluids.propulsion)
        loads = None
        if compute_loads:
            canard = aero.canard
            point_loads = (() if canard is None else
                           ((canard.station_m, canard.axial_force_n, canard.normal_force_n),))
            loads = self.loads.evaluate(
                atmosphere.q, atmosphere.Ma, alpha, forces['axial_aero'],
                forces["thrust_axial"], engine_on=engine_on,
                thrust_normal=forces["thrust_normal"], aerodynamic=not aero.ballistic_coast,
                point_loads=point_loads)
            for name in ("axial", "normal", "shear", "bending"):
                values = np.asarray(loads[name])
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Nonfinite {name} loads at t={kin.t}")
                result.load_peaks[name] = max(result.load_peaks.get(name, 0.0), float(np.max(np.abs(values))))
        if result.history is None:
            return None
        wind_x, wind_z = self._wind(kin.h)
        _, _, airspeed, gamma_air, _ = self.flight_kinematics(kin, wind_x=wind_x, wind_z=wind_z)
        state = dict(kinematics=kin, atmosphere=atmosphere, plant=plant, forces=forces,
                     alpha=alpha, mass_properties=mass, engine_on=engine_on, on_rail=on_rail,
                     wind=dict(wind_x=float(wind_x), wind_z=float(wind_z),
                               airspeed=float(airspeed), gamma_air=float(gamma_air)))
        if loads is not None:
            state["loads"] = loads
        return state

    def _record_twr(self, result, kin, twr, previous=None, previous_twr=None):
        result.max_twr = max(result.max_twr, twr)
        if result.rail_exit_time is not None:
            return
        sample_time = kin.t
        if not self.on_rail(kin.h):
            if previous is None or not self.on_rail(previous.h):
                result.rail_exit_time = kin.t
                return
            rail_end = self.rail_tip_altitude
            fraction = (rail_end - previous.h) / (kin.h - previous.h)
            sample_time = previous.t + fraction * (kin.t - previous.t)
            twr = previous_twr + fraction * (twr - previous_twr)
            result.rail_exit_time = sample_time
        if result.min_rail_twr is None or twr < result.min_rail_twr:
            result.min_rail_twr = twr
            result.min_rail_twr_time = sample_time

    def _record_extrema(self, result, atmosphere, aero, alpha, cg):
        result.max_aoa_deg = max(result.max_aoa_deg, abs(math.degrees(alpha)))
        result.max_q = max(result.max_q, float(atmosphere.q))
        diameter = self.cfg.get("vehicle", {}).get("OMLD")
        cp = getattr(aero, 'total_cp', aero.cp)
        if diameter is not None and math.isfinite(cp):
            stability = (cp - cg) / float(diameter)
            previous = result.min_stability_calibers
            result.min_stability_calibers = stability if previous is None else min(previous, stability)
            fraction = (cp - cg) / self.vehicle.length
            previous = result.min_stability_length_fraction
            result.min_stability_length_fraction = fraction if previous is None else min(previous, fraction)
            previous = result.max_stability_length_fraction
            result.max_stability_length_fraction = fraction if previous is None else max(previous, fraction)
        if getattr(self, 'roll_analysis', None) is not None and not aero.ballistic_coast and atmosphere.q > 0:
            cn_alpha, cm_alpha = self.aero.stability_derivatives(atmosphere.Ma, alpha, cg)
            scale = atmosphere.q * self.aero.reference_area
            normal_slope, moment_slope = cn_alpha * scale, cm_alpha * scale * diameter
            if aero.canard is not None:
                step, command = 1e-4, abs(aero.canard.deflection_rad)
                upper = self.canards.body_forces(atmosphere, alpha + step, command)[1]
                lower = self.canards.body_forces(atmosphere, alpha - step, command)[1]
                slope = (upper - lower) / (2 * step)
                normal_slope += slope
                moment_slope -= slope * (self.canards.station_m - cg)
            if abs(normal_slope) > 1e-12 * scale:
                aero.slope_stability_calibers = -moment_slope / (normal_slope * diameter)
                previous = result.min_slope_stability_calibers
                result.min_slope_stability_calibers = (aero.slope_stability_calibers if previous is None
                    else min(previous, aero.slope_stability_calibers))
            # Inertia is supplied at roll recording, so retain stiffness here.
            aero.pitch_stiffness_nm_rad = moment_slope

    def _record_roll(self, kin, evaluation, mass, on_rail):
        if self.roll_analysis is None:
            return
        atmosphere, aero, _, _ = evaluation
        if aero.ballistic_coast:
            atmosphere = replace(atmosphere, q=0.)
        stiffness = aero.pitch_stiffness_nm_rad
        frequency = math.sqrt(-stiffness / mass['Iyy']) / (2 * math.pi) if stiffness <= 0 else float('nan')
        aero.pitch_hz = frequency
        self.roll_analysis.sample(kin.t, atmosphere, self._flight_kinematics(kin)[2], mass,
                                 aero.canard, on_rail=on_rail, pitch_hz=frequency)

    @property
    def rail_tip_altitude(self):
        launch = self.cfg['launch']
        return float(launch['altitude']) + float(launch['rail_height']) * self.rail_sin

    def on_rail(self, altitude: float) -> bool:
        return altitude < self.rail_tip_altitude

    @staticmethod
    def engine_on(propulsion: PropulsionOut) -> bool:
        """Return whether combustion is active for aerodynamic deck selection."""

        return propulsion.mode == "combusting"
