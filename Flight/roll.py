import numpy as np

from Vehicle.utils.geometry import RocketAeroGeometry


def thrust_roll_moment(thrust, offset_y, angle_z, fin_forcing):
    # Resolve z-directed thrust and align its roll moment with fin cant.
    return np.sign(fin_forcing) * abs(offset_y * thrust * np.sin(angle_z))


def roll_moment(qbar, speed, p, area, diameter, fin_cant, cl_delta, cl_p,
                thrust_roll, external_roll):
    forcing = (thrust_roll + qbar * area * diameter * fin_cant * cl_delta
               + external_roll)
    damping = 0.0
    if speed > 1e-12:
        damping = qbar * area * diameter * cl_p * p * diameter / (2 * speed)
    return forcing + damping, forcing


def advance_roll(dt, p, phi, moment, inertia):
    # Advance roll rate and angle from the previous accepted moment.
    return p + moment * dt / inertia, phi + p * dt


class RollAnalysis:
    def __init__(self, diameter, inputs, aero):
        self.diameter = diameter
        self.area = np.pi * diameter**2 / 4
        self.inputs = inputs
        self.aero = aero

        candidate = aero.candidate
        inch = 0.0254
        root = candidate["root"] * inch
        tip = candidate["tip"] * inch
        span = candidate["span"] * inch
        length = candidate["boattail_length"] * inch
        r_le = diameter / 2 + (length - root) / length * (
            candidate["boattail_aft"] * inch / 2 - diameter / 2)
        r_te = candidate["boattail_aft"] * inch / 2
        self.fin_roll_lever_arm = 0.5 * (r_le + r_te + span)
        fin_geometry = RocketAeroGeometry(
            root_chord=root, tip_chord=tip, semi_span=span,
            body_radius=0.5 * (r_le + r_te),
        )
        self.roll_damping_shape = fin_geometry.roll_geometrical_constant
        self.roll_damping_interference = fin_geometry.roll_damping_interference_factor

        self.result = {
            "time": [], "p": [], "phi": [], "T_eq": [],
            "roll_moment": [], "open_roll_moment": [],
            "forcing_moment": [], "damping_moment": [],
            "canard_roll_moment": [], "pitch_hz": [],
        }
        self.previous_Ix = None

    def fin_coefficients(self, mach, alpha, cg, fin_cant):
        # Read fin and vehicle aero slopes at the current Mach and angle.
        cm_alpha = self.inputs.get("Cm_alpha")
        if cm_alpha is None:
            cm_alpha = self.aero.pitch_moment_derivative(mach, alpha, cg)

        cl_delta = self.inputs.get("Cl_delta")
        cl_p = self.inputs.get("Cl_p")
        if cl_delta is None or cl_p is None:
            fin_slope = self.aero.fin_normal_force_derivative(mach)
            if cl_delta is None:
                cl_delta = self.fin_roll_lever_arm * fin_slope / self.diameter
            if cl_p is None:
                cl_p = (-2 * self.roll_damping_interference * fin_slope *
                        np.cos(fin_cant) * self.roll_damping_shape /
                        (self.area * self.diameter**2))

        return cm_alpha, cl_delta, cl_p

    def evaluate(self, kin, atmosphere, vehicle, thrust=0):
        if self.result["time"]:
            dt = kin.t - self.result["time"][-1]
            p, phi = advance_roll(
                dt, self.result["p"][-1], self.result["phi"][-1],
                self.result["roll_moment"][-1], self.previous_Ix,
            )
        else:
            p = 0.0
            phi = 0.0

        fin_cant = self.inputs["fin_cant"]
        cm_alpha, cl_delta, cl_p = self.fin_coefficients(
            atmosphere.Ma, kin.alpha, vehicle.cg, fin_cant)

        thrust_roll = thrust_roll_moment(
            thrust, self.inputs.get("thrust_offset_y", 0.0),
            self.inputs.get("thrust_angle_z", 0.0), fin_cant * cl_delta)

        stiffness = atmosphere.q * self.area * self.diameter * cm_alpha
        pitch_hz = np.sqrt(-stiffness / vehicle.Iyy) / (2 * np.pi) if stiffness <= 0 else float("nan")

        speed = np.hypot(kin.vx, kin.vz)
        moment, forcing = roll_moment(
            atmosphere.q, speed, p, self.area, self.diameter, fin_cant,
            cl_delta, cl_p, thrust_roll,
            self.inputs.get("M_roll_external", 0.0),
        )
        return {
            "p": p, "phi": phi, "open_moment": moment,
            "forcing_moment": forcing,
            "T_eq": thrust_roll, "pitch_hz": pitch_hz,
        }

    def sample(self, kin, atmosphere, vehicle, thrust=0, control_moment=0):
        state = self.evaluate(kin, atmosphere, vehicle, thrust)
        p = state["p"]
        moment = state["open_moment"] + control_moment

        self.result["time"].append(kin.t)
        self.result["p"].append(p)
        self.result["phi"].append(state["phi"])
        self.result["T_eq"].append(state["T_eq"])
        self.result["roll_moment"].append(moment)
        self.result["open_roll_moment"].append(state["open_moment"])
        self.result["forcing_moment"].append(state["forcing_moment"])
        self.result["damping_moment"].append(
            state["open_moment"] - state["forcing_moment"])
        self.result["canard_roll_moment"].append(control_moment)
        self.result["pitch_hz"].append(state["pitch_hz"])
        self.previous_Ix = float(vehicle.Ixx)
