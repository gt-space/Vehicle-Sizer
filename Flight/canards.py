from dataclasses import dataclass
from math import pi, radians, sqrt

import numpy as np
from scipy.special import ellipe


@dataclass
class CanardGeometry:
    area: float
    aspect_ratio: float
    moment_arm: float
    body_factor: float


@dataclass
class CanardForces:
    deflection_rad: float = 0.0
    open_roll_moment_nm: float = 0.0
    control_roll_moment_nm: float = 0.0
    remaining_roll_moment_nm: float = 0.0
    max_roll_moment_nm: float = 0.0
    authority_ratio: float = float("inf")
    saturated: bool = False
    drag_n: float = 0.0
    neutral_drag_n: float = 0.0
    normal_force_n: float = 0.0
    pitch_moment_nm: float = 0.0
    cn_alpha: float = 0.0


class CanardAerodynamics:
    def __init__(self, cfg):
        self.cfg = cfg

    def surface_geometry(self, root, radius):
        # Size the paired triangular canards from a 74-degree leading edge.
        span = root / np.tan(radians(74.0))
        standoff = 0.020
        area = 0.5 * root * span
        ybar = 4 * span / (3 * pi)
        attached = 1 + radius / (span + radius)
        gap = max(0.0, 1 - standoff / span)
        return CanardGeometry(area, 2 * span**2 / area,
                              radius + standoff + ybar,
                              1 + (attached - 1) * gap)

    def polhamus(self, mach):
        mach_table = [0.60, 1.20, 2.00, 2.36, 2.80]
        kp_table = [1.539, 1.838, 1.484, 1.325, 1.210]
        kv_table = [2.869, 1.766, 0.722, 0.709, 0.521]
        bounded = np.clip(mach, mach_table[0], mach_table[-1])
        return np.interp(bounded, mach_table, kp_table), np.interp(
            bounded, mach_table, kv_table)

    def finite_wing_factor(self, geometry, mach):
        """Triangular-wing conical lift: C_Nalpha=pi*AR/(2*E(sqrt(1-m^2)))
        for m=beta*cot(74 deg)<1, and 4/beta for m>=1; NACA TR-1050,
        TR-970 (pointed-tip limit), and TN-1183 (supersonic leading edge).
        """
        beta = np.sqrt(mach**2 - 1)
        m = beta / np.tan(radians(74.0))
        elliptic_e = ellipe(np.maximum(0, 1 - m**2))
        return np.where(m < 1, pi * beta * geometry.aspect_ratio /
                        (8 * elliptic_e), 1.0)

    def canard_lift(self, geometry, mach, angle):
        mach, angle = np.broadcast_arrays(np.asarray(mach, dtype=float),
                                           np.asarray(angle, dtype=float))
        kp, kv = self.polhamus(mach)
        lift = np.asarray(kp * np.sin(angle) * np.cos(angle)**2 +
                          kv * np.cos(angle) * np.sin(angle) * np.abs(np.sin(angle))).copy()
        high = mach > 2.8
        if np.any(high):
            hypersonic = self.shock_expansion_cn(mach[high], angle[high])
            hypersonic *= self.finite_wing_factor(
                geometry, mach[high]) * np.cos(angle[high])
            blend = np.clip((mach[high] - 2.8) / 1.2, 0, 1)
            lift[high] = (1 - blend) * lift[high] + blend * hypersonic
        return lift

    def roll_coefficient(self, geometry, mach, command, body_alpha):
        command, mach, body_alpha = np.broadcast_arrays(
            np.asarray(command, dtype=float), np.asarray(mach, dtype=float),
            np.asarray(body_alpha, dtype=float))
        fractions = np.linspace(0, 1, 9).reshape((-1,) + (1,) * command.ndim)
        incidence = fractions * np.abs(body_alpha)
        positive = command + incidence
        negative = command - incidence
        cp = self.canard_lift(geometry, mach, positive) / np.cos(positive)
        cm = self.canard_lift(geometry, mach, negative) / np.cos(negative)
        coefficient = np.min(0.5 * (cp + cm) * np.cos(command), axis=0)
        return np.maximum(coefficient, 0)

    def drag(self, geometry, mach, command, body_alpha, count, q):
        area = geometry.area
        cd0 = self.cfg.get("cd0", 0.01)
        oswald = self.cfg.get("oswald", 0.8)
        deflection_drag_multiplier = self.cfg.get("deflection_drag_multiplier", 1.0)
        incidence = np.linspace(0, 1, 9) * abs(body_alpha)
        angles = np.concatenate((command + incidence, command - incidence,
                                 incidence, -incidence))
        lift = geometry.body_factor * self.canard_lift(geometry, mach, angles)
        subsonic = lift**2 / (pi * oswald * geometry.aspect_ratio)
        supersonic = np.abs(lift * np.tan(angles))
        blend = np.clip((mach - 0.8) / 0.4, 0, 1)
        coefficients = cd0 + (1 - blend) * subsonic + blend * supersonic
        drag = 0.5 * count * q * area * np.max(
            coefficients[:9] + coefficients[9:18])
        neutral = 0.5 * count * q * area * np.max(
            coefficients[18:27] + coefficients[27:])
        drag = neutral + deflection_drag_multiplier * (drag - neutral)
        return drag, neutral

    def shock_expansion_cn(self, mach, angle):
        gamma = 1.4
        angle_abs = np.abs(angle)
        mach2 = mach**2
        mach_angle = np.arcsin(1 / mach)
        sin_beta2 = ((gamma + 1) * mach2 / 4 - 1 + np.sqrt(
            (gamma + 1) * ((gamma + 1) * mach2**2 / 16 +
                           (gamma - 1) * mach2 / 2 + 1))) / (gamma * mach2)
        beta_max = np.arcsin(np.sqrt(np.clip(sin_beta2, 0, 1)))
        theta_max = self.shock_turn(mach, beta_max, gamma)
        valid = (angle_abs <= theta_max) & (mach > 1)
        low = mach_angle.copy()
        high = beta_max.copy()
        for _ in range(45):
            middle = (low + high) / 2
            lower_side = self.shock_turn(mach, middle, gamma) < angle_abs
            low = np.where(lower_side, middle, low)
            high = np.where(lower_side, high, middle)
        beta = (low + high) / 2
        compression = 1 + 2 * gamma / (gamma + 1) * (
            mach2 * np.sin(beta)**2 - 1)
        target_nu = self.pm_angle(mach, gamma) + angle_abs
        low = mach.copy()
        high = np.full(mach.shape, 1000.0)
        valid &= target_nu < self.pm_angle(high, gamma)
        for _ in range(50):
            middle = (low + high) / 2
            lower_side = self.pm_angle(middle, gamma) < target_nu
            low = np.where(lower_side, middle, low)
            high = np.where(lower_side, high, middle)
        expanded_mach = (low + high) / 2
        expansion = ((1 + (gamma - 1) * mach2 / 2) /
                     (1 + (gamma - 1) * expanded_mach**2 / 2))**(
                         gamma / (gamma - 1))
        cn = np.sign(angle) * (compression - expansion) / (0.5 * gamma * mach2)
        cn = np.where(angle_abs < 1e-12, 0, cn)
        return np.where(valid, cn, np.nan)

    @staticmethod
    def shock_turn(mach, beta, gamma):
        return np.arctan(2 / np.tan(beta) * (mach**2 * np.sin(beta)**2 - 1) /
                         (mach**2 * (gamma + np.cos(2 * beta)) + 2))

    @staticmethod
    def pm_angle(mach, gamma):
        x = np.sqrt(np.maximum(mach**2 - 1, 0))
        return (sqrt((gamma + 1) / (gamma - 1)) *
                np.arctan(sqrt((gamma - 1) / (gamma + 1)) * x) - np.arctan(x))


class CanardSystem:
    def __init__(self, config, diameter, vehicle):
        self.config = config
        self.aero = CanardAerodynamics(config.get("aero", {}))
        self.geometry = self.aero.surface_geometry(config["root_chord_m"], diameter / 2)
        self.reference_area = np.pi * diameter**2 / 4
        self.count = 2
        self.stability_on = config.get("canard_stablity_on", True)
        self.station_m = config.get("station_m")
        if self.station_m is None:
            for section in vehicle.sections:
                if "canards" in section.mass_inputs.get("masses", {}):
                    # Locate the triangular canard's area centroid from the housing front.
                    self.station_m = section.start_station + 2 * config["root_chord_m"] / 3
                    break

    def stability_effect(self, kin, atmosphere, cg):
        if not self.stability_on:
            return 0.0, 0.0, 0.0

        alpha = kin.alpha
        step = 0.0001
        angles = np.array([alpha, alpha + step, alpha - step])
        lift = self.aero.canard_lift(self.geometry, atmosphere.Ma, angles)

        area = self.count * self.geometry.area * self.geometry.body_factor
        normal = atmosphere.q * area * lift[0] * np.cos(alpha)

        upper = lift[1] * np.cos(alpha + step)
        lower = lift[2] * np.cos(alpha - step)
        cn_alpha = area * (upper - lower) / (2 * step * self.reference_area)

        pitch = -normal * (self.station_m - cg)
        return float(normal), float(pitch), float(cn_alpha)

    def command_deflection(self, mach, alpha, open_moment, torque_scale, active):
        # Select the deflection that opposes the current roll forcing moment.
        angle_limit = radians(self.config["max_deflection_deg"])
        if mach <= 2.8:
            lift_limit = radians(self.config.get("max_measured_angle_deg", 22.0))
        else:
            lift_limit = radians(self.config.get("max_extension_angle_deg", 15.0))
        angle_limit = min(angle_limit, max(lift_limit - abs(alpha), 0))
        maximum_lift = float(self.aero.roll_coefficient(
            self.geometry, mach, angle_limit, alpha))
        maximum_torque = torque_scale * maximum_lift
        demand = abs(open_moment)
        command = 0.0
        saturated = False
        if active and demand > 0:
            if maximum_torque > 0:
                angles = np.linspace(0, angle_limit, 240)
                lift_curve = self.aero.roll_coefficient(
                    self.geometry, mach, angles, alpha)
                required = demand / torque_scale
                if required >= maximum_lift:
                    command = angle_limit
                    saturated = required > maximum_lift
                else:
                    command = float(np.interp(required, lift_curve, angles))
            else:
                saturated = True
        return command, maximum_torque, saturated

    def evaluate(self, kin, atmosphere, forcing_moment, open_roll_moment, active, cg):
        q = atmosphere.q
        mach = atmosphere.Ma
        alpha = kin.alpha
        multiplier = self.config.get("net_roll_multiplier", 1.0)
        torque_scale = (self.count * q * self.geometry.area *
                        self.geometry.body_factor * self.geometry.moment_arm *
                        multiplier)
        command, maximum_torque, saturated = self.command_deflection(
            mach, alpha, forcing_moment, torque_scale, active)
        control = (-np.sign(forcing_moment) * torque_scale *
                   float(self.aero.roll_coefficient(self.geometry, mach, command, alpha)))
        if not active:
            control = 0.0
        drag, neutral = self.aero.drag(
            self.geometry, mach, command, alpha, self.count, q)
        normal, pitch, cn_alpha = self.stability_effect(kin, atmosphere, cg)
        demand = abs(forcing_moment)
        authority = maximum_torque / demand if demand > 0 else float("inf")
        return CanardForces(
            deflection_rad=-np.sign(forcing_moment) * command,
            open_roll_moment_nm=open_roll_moment,
            control_roll_moment_nm=control,
            remaining_roll_moment_nm=open_roll_moment + control,
            max_roll_moment_nm=maximum_torque,
            authority_ratio=authority,
            saturated=saturated,
            drag_n=drag,
            neutral_drag_n=neutral,
            normal_force_n=normal,
            pitch_moment_nm=pitch,
            cn_alpha=cn_alpha,
        )
