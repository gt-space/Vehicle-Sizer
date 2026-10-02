import numpy as np

from Vehicle.utils.aoa import (
    propagate_roll_modes,
    roll_moment,
    roll_pitch_yaw_upper_envelope,
    thrust_moments,
    transverse_modes,
)
from Vehicle.utils.geometry import RocketAeroGeometry


class RollAnalysis:
    def __init__(self, diameter, inputs, aero=None, fin_count=4):
        self.diameter = diameter
        self.area = np.pi * diameter**2 / 4
        self.aero = aero
        self.fin_count = fin_count
        self.mode = inputs.get("mode", "batch")
        self.inputs = inputs.copy()
        self.inputs.pop("mode", None)
        self.inputs.pop("Iz", None)
        if "Cl_delta" not in self.inputs or "Cl_p" not in self.inputs:
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
            self.roll_damping_interference = (
                fin_geometry.roll_damping_interference_factor
            )
        self.samples = []
        self._records = {}
        for key in (
            "p", "phi", "T_eq", "roll_moment", "R1", "R2",
            "lambda_1", "lambda_2", "omega_1",
            "omega_2", "diagnostic_modes", "alpha_trim", "alpha_upper",
            "q_alpha_upper",
        ):
            self._records[key] = []

    def sample(self, kin, atmosphere, vehicle, thrust=0):
        index = len(self.samples)
        fin_cant = self.inputs["fin_cant"]
        if np.ndim(fin_cant):
            fin_cant = fin_cant[index]

        # Read the static pitch and fin slopes at this flight condition.
        if "Cm_alpha" in self.inputs:
            cm_alpha = self.inputs["Cm_alpha"]
            if np.ndim(cm_alpha):
                cm_alpha = cm_alpha[index]
        else:
            cm_alpha = self.aero.pitch_moment_derivative(
                atmosphere.Ma, kin.alpha, vehicle.cg
            )

        if "Cl_delta" in self.inputs:
            cl_delta = self.inputs["Cl_delta"]
            if np.ndim(cl_delta):
                cl_delta = cl_delta[index]
        else:
            fin_slope = self.aero.fin_normal_force_derivative(atmosphere.Ma)
            cl_delta = self.fin_roll_lever_arm * fin_slope / self.diameter

        if "Cl_p" in self.inputs:
            cl_p = self.inputs["Cl_p"]
            if np.ndim(cl_p):
                cl_p = cl_p[index]
        else:
            fin_slope = self.aero.fin_normal_force_derivative(atmosphere.Ma)
            cl_p = (-2 * self.roll_damping_interference * fin_slope *
                    np.cos(fin_cant) * self.roll_damping_shape /
                    (self.area * self.diameter**2))

        if "CN_alpha" in self.inputs:
            cn_alpha = self.inputs["CN_alpha"]
            if np.ndim(cn_alpha):
                cn_alpha = cn_alpha[index]
        else:
            cn_alpha = self.aero.normal_force_derivative(atmosphere.Ma, kin.alpha)

        cg_offset = self.inputs.get("cg_offset", 0)
        Gamma = self.inputs.get("Gamma", 0)
        cl1 = self.inputs.get("Cl1", 0)
        if np.ndim(cg_offset):
            cg_offset = cg_offset[index]
        if np.ndim(Gamma):
            Gamma = Gamma[index]
        if np.ndim(cl1):
            cl1 = cl1[index]

        # Resolve nozzle thrust about the moving center of mass.
        nozzle_x = self.inputs.get("thrust_point_x", vehicle.engine_end_station)
        nozzle_y = self.inputs.get("thrust_offset_y", 0)
        nozzle_z = self.inputs.get("thrust_offset_z", 0)
        angle_y = self.inputs.get("thrust_angle_y", 0)
        angle_z = self.inputs.get("thrust_angle_z", 0)
        # Vehicle stations increase toward the tail; thrust points forward.
        arm_x = vehicle.cg - nozzle_x
        arm_y = nozzle_y - cg_offset * np.cos(Gamma)
        arm_z = nozzle_z - cg_offset * np.sin(Gamma)
        T_eq, pitch_thrust, yaw_thrust = thrust_moments(
            thrust, arm_x, arm_y, arm_z, angle_y, angle_z)
        if "T_eq" in self.inputs:
            T_eq = self.inputs["T_eq"]
            if np.ndim(T_eq):
                T_eq = T_eq[index]

        pitch_external = self.inputs.get("M_pitch_external", 0)
        yaw_external = self.inputs.get("M_yaw_external", 0)
        roll_external = self.inputs.get("M_roll_external", 0)
        if np.ndim(pitch_external):
            pitch_external = pitch_external[index]
        if np.ndim(yaw_external):
            yaw_external = yaw_external[index]
        if np.ndim(roll_external):
            roll_external = roll_external[index]
        pitch_external += pitch_thrust
        yaw_external += yaw_thrust

        current = (kin.t, atmosphere.q, np.hypot(kin.vx, kin.vz),
                   float(vehicle.Ixx), float(vehicle.Iyy), float(vehicle.Izz),
                   cm_alpha, cl_delta, cl_p, cn_alpha, T_eq,
                   pitch_external, yaw_external, roll_external)
        print(f'Ixx: {vehicle.Ixx}')
        self.samples.append(current)
        if self.mode != "inline":
            return

        if index == 0:
            p = self.inputs["p0"]
            phi = self.inputs.get("phi0", 0)
            modes = np.array([self.inputs["R1_0"], self.inputs["R2_0"]])
        else:
            previous = self.samples[-2]
            old_modes = np.array([self._records["R1"][-1],
                                  self._records["R2"][-1]])
            p, phi, modes = propagate_roll_modes(
                kin.t - previous[0], self._records["p"][-1],
                self._records["phi"][-1], old_modes, self._previous_growth,
                self._records["roll_moment"][-1], previous[3],
            )

        coeff = {"Cm_alpha": cm_alpha, "Cl_delta": cl_delta,
                 "Cl_p": cl_p, "fin_cant": fin_cant}
        cm_q = self.inputs["Cm_q"]
        if np.ndim(cm_q):
            cm_q = cm_q[index]
        coeff["Cm_q"] = cm_q
        coeff["M_pitch_external"] = pitch_external
        coeff["M_yaw_external"] = yaw_external
        roll_rate = self.inputs.get("roll_rate")
        if roll_rate is not None:
            p = roll_rate[index]
        qbar = current[1]
        speed = current[2]
        Ix = current[3]
        Iy = current[4]
        Iz = current[5]
        growth, frequency, trim, diagnostic = transverse_modes(
            qbar, speed, Ix, Iy, Iz, self.area, self.diameter,
            coeff["Cm_alpha"], coeff["Cm_q"], p,
            coeff["M_pitch_external"], coeff["M_yaw_external"],
        )
        moment = roll_moment(
            qbar, speed, p, self.area, self.diameter, fin_cant,
            cl_delta, cl_p, cl1, self.fin_count, phi, cg_offset,
            cn_alpha, trim, Gamma, T_eq, roll_external)
        alpha = trim + np.abs(modes).sum()
        pitch_stiffness = qbar * self.area * self.diameter * cm_alpha
        pitch_hz = float("nan")
        if pitch_stiffness <= 0:
            pitch_hz = np.sqrt(-pitch_stiffness / Iy) / (2 * np.pi)
        # Show the roll rate, pitch frequency and modal AoA at this step.
        print(f"Roll analysis: t={kin.t:.3f} s, roll rate={p / (2 * np.pi):.6g} Hz, "
              f"pitch natural frequency={pitch_hz:.6g} Hz, "
              f"AoA={np.rad2deg(alpha):.6g} deg", flush=True)
        values = (p, phi, T_eq, moment, modes[0], modes[1], growth[0], growth[1],
                  frequency[0], frequency[1], diagnostic, trim, alpha,
                  qbar * alpha)
        for key, value in zip(self._records, values):
            self._records[key].append(value)
        self._previous_growth = growth

    def evaluate(self):
        time, qbar, speed, Ix, Iy, Iz, Cm_alpha, Cl_delta, Cl_p, CN_alpha, T_eq, Mp, My, Mr = (
            np.asarray(self.samples).T
        )
        if self.mode != "inline":
            inputs = self.inputs.copy()
            inputs.pop("Cm_alpha", None)
            inputs.pop("Cl_delta", None)
            for key in ("Cl_p", "CN_alpha", "T_eq", "M_pitch_external",
                        "M_yaw_external", "M_roll_external", "cg_offset",
                        "Gamma", "Cl1", "phi0", "thrust_point_x",
                        "thrust_offset_y", "thrust_offset_z",
                        "thrust_angle_y", "thrust_angle_z"):
                inputs.pop(key, None)
            return roll_pitch_yaw_upper_envelope(
                time, qbar, speed, Ix, Iy, Iz,
                A=self.area, d=self.diameter, Cm_alpha=Cm_alpha,
                Cl_delta=Cl_delta, Cl_p=Cl_p, CN_alpha=CN_alpha,
                T_eq=T_eq, M_pitch_external=Mp, M_yaw_external=My,
                M_roll_external=Mr, Cl1=self.inputs.get("Cl1", 0),
                N=self.fin_count, cg_offset=self.inputs.get("cg_offset", 0),
                Gamma=self.inputs.get("Gamma", 0), phi0=self.inputs.get("phi0", 0),
                **inputs,
            )
        result = {}
        for key, value in self._records.items():
            result[key] = np.asarray(value)
        result["time"] = time
        ia = np.argmax(result["alpha_upper"])
        iq = np.argmax(result["q_alpha_upper"])
        result.update(
            peak_alpha_rad=result["alpha_upper"][ia],
            peak_alpha_deg=np.rad2deg(result["alpha_upper"][ia]),
            peak_alpha_time=time[ia],
            peak_q_alpha=result["q_alpha_upper"][iq],
            peak_q_alpha_time=time[iq],
        )
        return result
