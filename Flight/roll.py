"""Auxiliary roll forcing and accepted-step integration; no feedback into planar attitude."""
import numpy as np


def roll_geometry(root, tip, span, radius):
    """Fin chord/radius integral and fin-body damping interference."""
    shape = span / 12 * ((root + 3 * tip) * span**2
                         + 4 * (root + 2 * tip) * span * radius
                         + 6 * (root + tip) * radius**2)
    tau, taper = (span + radius) / radius, tip / root
    numerator = (tau - taper) / tau - (1 - taper) / (tau - 1) * np.log(tau)
    denominator = ((tau + 1) * (tau - taper) / 2
                   - (1 - taper) * (tau**3 - 1) / (3 * (tau - 1)))
    return shape, 1 + numerator / denominator


class RollAnalysis:
    def __init__(self, diameter, inputs, aero):
        self.diameter, self.inputs, self.aero = diameter, inputs, aero
        self.area = np.pi * diameter**2 / 4
        self.cant = float(inputs.get("fin_cant", 0.))
        candidate = aero.candidate
        root, tip, span = (candidate[key] * .0254 for key in ("root", "tip", "span"))
        length, aft_radius = candidate["boattail_length"] * .0254, candidate["boattail_aft"] * .0254 / 2
        radius = diameter / 2
        leading_radius = radius + (length - root) / length * (aft_radius - radius) if length else radius
        self.lever = .5 * (leading_radius + aft_radius + span)
        self.shape, self.interference = roll_geometry(root, tip, span, radius)
        self.reset()

    def reset(self):
        self.result = {key: [] for key in ("time", "p", "phi", "roll_moment", "open_roll_moment",
            "forcing_moment", "damping_moment", "canard_roll_moment", "pitch_hz")}
        self.previous_Ixx = None

    def coefficients(self, mach):
        delta, damping = self.inputs.get("Cl_delta"), self.inputs.get("Cl_p")
        if delta is None or damping is None:
            slope = self.aero.fin_normal_force_derivative(mach)
            if delta is None:
                delta = self.lever * slope / self.diameter
            if damping is None:
                damping = (-2 * self.interference * slope * np.cos(self.cant) * self.shape
                           / (self.area * self.diameter**2))
        return delta, damping

    def forcing(self, atmosphere):
        delta, _ = self.coefficients(atmosphere.Ma)
        fin_forcing = self.cant * delta
        return float(atmosphere.q * self.area * self.diameter * fin_forcing
                     + self.inputs.get("M_roll_external", 0.))

    def sample(self, time, atmosphere, airspeed, mass, canard=None, *,
               on_rail=False, pitch_hz=float("nan")):
        """Advance from the previous accepted moment; replace a same-time event side."""
        result = self.result
        p = phi = 0.
        if result["time"]:
            dt = time - result["time"][-1]
            if dt < 0:
                raise ValueError("Roll samples must be chronological")
            p = result["p"][-1] + dt * result["roll_moment"][-1] / self.previous_Ixx
            phi = result["phi"][-1] + dt * result["p"][-1]
            if dt == 0:
                for values in result.values():
                    values.pop()
        if on_rail:
            p = phi = 0.
        _, coefficient = self.coefficients(atmosphere.Ma)
        damping = (atmosphere.q * self.area * self.diameter * coefficient
                   * p * self.diameter / (2 * airspeed)) if airspeed > 1e-12 else 0.
        forcing = self.forcing(atmosphere)
        control = canard.control_roll_moment_nm if canard is not None else 0.
        moment = 0. if on_rail else forcing + damping + control
        for key, value in zip(result, (time, p, phi, moment, forcing + damping,
                                      forcing, damping, control, pitch_hz)):
            result[key].append(float(value))
        self.previous_Ixx = mass["Ixx"]
