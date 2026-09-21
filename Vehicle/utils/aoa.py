"""Conservative AoA envelope from roll_pitch_yaw_coupling_handoff.docx.

This simplified principal-axis model preserves unequal transverse inertias.
It omits translational normal-force and Cm_dot_alpha coupling, discards modal
phase, and sorts modes by frequency (labels can swap at crossings). It is not
a validated predictor or a phase-resolved attitude solution. Initial modal
amplitudes and missing aerodynamic derivatives must be supplied by the caller.
"""

import numpy as np


def Cl_delta(N, span, r_LE, r_TE, CN_alpha_1, L_r):
    """Fin-cant derivative [1/rad], using the fin-center/slanted-body model.

    Lengths are metres; CN_alpha_1 is a single-fin derivative referenced to
    the same area as the envelope coefficients. It may be a Mach-evaluated
    history. Local leading/trailing-edge body radii are explicit inputs.
    """
    return N * (0.5 * (r_LE + r_TE + span)) * np.asarray(CN_alpha_1) / L_r


def roll_pitch_yaw_upper_envelope(
    time, qbar, V, Ix, Iy, Iz, A, d, Cm_alpha, Cm_q, Cl_delta, Cl_p,
    fin_cant, p0, R1_0, R2_0, M_pitch_external=0, M_yaw_external=0,
    M_roll_external=0, *, roll_rate=None,
):
    """Return AoA/q·AoA histories, modal diagnostics and peaks with timestamps.

    Units: seconds, Pa, m/s (speed), kg*m², m², metres, radians, rad/s,
    and N*m. q_alpha_upper is Pa*rad. Coefficients, inertias, cant and
    moments accept scalars or 1D histories; time/qbar/V must be matching
    nonempty 1D arrays, with strictly increasing time and positive inertias.
    Geometry.fin_cant is in degrees: convert it with np.deg2rad.

    Supply roll_rate to use an externally integrated p(t); otherwise roll
    uses explicit Euler. Modal amplitudes use left-endpoint growth rates,
    consistent with R[k+1] = R[k]*exp(lambda[k]*dt) in the handoff equations.
    Modes are frequency sorted when oscillatory. For non-oscillatory cases,
    the two largest distinct-mode growth rates provide diagnostic amplitudes;
    this fallback is not a rigorous modal decomposition.

    Static trim is zero where stiffness vanishes, as in the handoff; external
    transverse forcing there is not dynamically propagated. Changing trim
    and inertia do not excite modes in this simplified formulation.
    """
    time = np.asarray(time, dtype=float)
    n = time.size
    qbar = np.asarray(qbar)
    V = np.asarray(V)
    Ix, Iy, Iz, Cm_alpha, Cm_q, Cl_delta, Cl_p, fin_cant, Mp, My, Mr = [
        np.broadcast_to(value, (n,)) for value in (
            Ix, Iy, Iz, Cm_alpha, Cm_q, Cl_delta, Cl_p, fin_cant,
            M_pitch_external, M_yaw_external, M_roll_external)
    ]
    p = np.zeros(n) if roll_rate is None else np.array(roll_rate, dtype=float)
    if roll_rate is None:
        p[0] = p0
    amplitudes = np.zeros((n, 2))
    amplitudes[0] = [R1_0, R2_0]
    growth, frequency = np.zeros((n, 2)), np.zeros((n, 2))
    trim = np.zeros(n)
    diagnostic = np.zeros(n, dtype=bool)
    for i in range(n):
        stiffness = qbar[i] * A * d * Cm_alpha[i]
        speed_factor = d / (2 * V[i]) if V[i] > 1e-12 else 0.0
        damping = qbar[i] * A * d * Cm_q[i] * speed_factor
        matrix = np.array([
            [0, 0, 1, 0], [0, 0, 0, 1],
            [stiffness / Iy[i], 0, damping / Iy[i], -(Ix[i] - Iz[i]) * p[i] / Iy[i]],
            [0, stiffness / Iz[i], (Ix[i] - Iy[i]) * p[i] / Iz[i], damping / Iz[i]],
        ])
        eigenvalues = np.linalg.eigvals(matrix)
        oscillatory = eigenvalues[eigenvalues.imag > 1e-10]
        if len(oscillatory) == 2:
            modes = oscillatory[np.argsort(oscillatory.imag)]
        else:
            diagnostic[i] = True
            # Keep one representative per conjugate pair, so a decaying pair
            # cannot hide a real unstable root (the handoff fallback could).
            candidates = np.concatenate((oscillatory, eigenvalues[np.abs(eigenvalues.imag) <= 1e-10]))
            modes = candidates[np.argsort(-candidates.real, kind="stable")[:2]]
        growth[i], frequency[i] = modes.real, np.abs(modes.imag)
        if abs(stiffness) > 1e-12:
            trim[i] = np.hypot(Mp[i], My[i]) / abs(stiffness)
        if i < n - 1:
            dt = time[i + 1] - time[i]
            amplitudes[i + 1] = amplitudes[i] * np.exp(growth[i] * dt)
            if roll_rate is None:
                moment = qbar[i] * A * d * (fin_cant[i] * Cl_delta[i] + Cl_p[i] * p[i] * speed_factor)
                p[i + 1] = p[i] + (moment + Mr[i]) / Ix[i] * dt
    alpha = trim + np.abs(amplitudes).sum(axis=1)
    qalpha = qbar * alpha
    ia, iq = np.argmax(alpha), np.argmax(qalpha)
    return dict(time=time.copy(), p=p, R1=amplitudes[:, 0], R2=amplitudes[:, 1],
                lambda_1=growth[:, 0], lambda_2=growth[:, 1],
                omega_1=frequency[:, 0], omega_2=frequency[:, 1],
                diagnostic_modes=diagnostic, alpha_trim=trim, alpha_upper=alpha,
                q_alpha_upper=qalpha, peak_alpha_rad=alpha[ia],
                peak_alpha_deg=np.rad2deg(alpha[ia]), peak_alpha_time=time[ia],
                peak_q_alpha=qalpha[iq], peak_q_alpha_time=time[iq])


def aoa_from_flight_history(flight_history, geometry, *, Iz, **inputs):
    """Adapt FlightSim.run() snapshots without changing the trajectory solver.

    Required inputs: Cm_alpha, Cm_q, Cl_delta, Cl_p, p0, R1_0, R2_0.
    Missing derivatives and Iz are explicit integration hooks; no symmetry or
    coefficient values are assumed. Optional envelope arguments also pass
    through. Compute Cl_delta with the helper above when a single-fin slope
    and local root radii are available. Existing kin.w is not automatically
    used: FlightSim currently carries it unchanged rather than integrating it.
    """
    rows = list(flight_history)
    return roll_pitch_yaw_upper_envelope(
        time=[row["kinematics"].t for row in rows],
        qbar=[row["atmosphere"].q for row in rows],
        V=[abs(row["kinematics"].v) for row in rows],
        Ix=[row["mass_properties"]["Ixx"] for row in rows],
        Iy=[row["mass_properties"]["Iyy"] for row in rows], Iz=Iz,
        A=geometry.reference_area, d=geometry.diameter,
        fin_cant=np.deg2rad(geometry.fin_cant), **inputs,
    )
