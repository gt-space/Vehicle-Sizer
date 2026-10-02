import numpy as np


def Cl_delta(N, span, r_LE, r_TE, CN_alpha_1, L_r):
    # Use the fin-center distance as the roll lever arm.
    return N * (0.5 * (r_LE + r_TE + span)) * np.asarray(CN_alpha_1) / L_r


def transverse_modes(qbar, V, Ix, Iy, Iz, A, d, Cm_alpha, Cm_q, p,
                     M_pitch_external=0, M_yaw_external=0):
    # Build the pitch-yaw motion matrix at this flight condition.
    stiffness = qbar * A * d * Cm_alpha
    speed_factor = d / (2 * V) if V > 1e-12 else 0.0
    damping = qbar * A * d * Cm_q * speed_factor
    matrix = np.array([
        [0, 0, 1, 0], [0, 0, 0, 1],
        [stiffness / Iy, 0, damping / Iy, -(Ix - Iz) * p / Iy],
        [0, stiffness / Iz, (Ix - Iy) * p / Iz, damping / Iz],
    ])
    eigenvalues = np.linalg.eigvals(matrix)
    oscillatory = eigenvalues[eigenvalues.imag > 1e-10]
    diagnostic = len(oscillatory) != 2
    if diagnostic:
        candidates = np.concatenate((
            oscillatory, eigenvalues[np.abs(eigenvalues.imag) <= 1e-10]
        ))
        modes = candidates[np.argsort(-candidates.real, kind="stable")[:2]]
    else:
        modes = oscillatory[np.argsort(oscillatory.imag)]
    trim = (np.hypot(M_pitch_external, M_yaw_external) / abs(stiffness)
            if abs(stiffness) > 1e-12 else 0.0)
    return modes.real, np.abs(modes.imag), trim, diagnostic


def thrust_moments(thrust, arm_x, arm_y, arm_z, angle_y, angle_z):
    # Resolve thrust and take the moment about the center of mass.
    force_x = thrust * np.cos(angle_y) * np.cos(angle_z)
    force_y = thrust * np.sin(angle_y)
    force_z = thrust * np.sin(angle_z)
    roll = arm_y * force_z - arm_z * force_y
    pitch = arm_z * force_x - arm_x * force_z
    yaw = arm_x * force_y - arm_y * force_x
    return roll, pitch, yaw


def roll_moment(qbar, V, p, A, d, fin_cant, Cl_delta, Cl_p,
                Cl1, N, phi, cg_offset, CN_alpha, alpha_trim, Gamma,
                T_eq=0, M_roll_external=0):
    # Add fin cant, damping, fin asymmetry, CG offset and thrust torque.
    speed_factor = d / (2 * V) if V > 1e-12 else 0.0
    coefficient = fin_cant * Cl_delta + Cl_p * p * speed_factor
    coefficient -= Cl1 * np.sin(N * phi)
    coefficient -= (cg_offset / d) * CN_alpha * alpha_trim * np.sin(phi - Gamma)
    return T_eq + qbar * A * d * coefficient + M_roll_external


def propagate_roll_modes(dt, p, phi, amplitudes, growth, moment, Ix):
    # Integrate roll rate, roll angle and the two modal amplitudes.
    next_p = p + moment / Ix * dt
    next_phi = phi + p * dt
    next_amplitudes = np.asarray(amplitudes) * np.exp(np.asarray(growth) * dt)
    return next_p, next_phi, next_amplitudes


def roll_pitch_yaw_upper_envelope(
    time, qbar, V, Ix, Iy, Iz, A, d, Cm_alpha, Cm_q, Cl_delta, Cl_p,
    fin_cant, p0, R1_0, R2_0, M_pitch_external=0, M_yaw_external=0,
    M_roll_external=0, roll_rate=None, T_eq=0, Cl1=0, N=4,
    cg_offset=0, Gamma=0, CN_alpha=0, phi0=0,
):
    # Put the varying flight quantities on the same time grid.
    time = np.asarray(time, dtype=float)
    n = time.size
    qbar = np.asarray(qbar)
    V = np.asarray(V)
    Ix = np.broadcast_to(Ix, (n,))
    Iy = np.broadcast_to(Iy, (n,))
    Iz = np.broadcast_to(Iz, (n,))
    Cm_alpha = np.broadcast_to(Cm_alpha, (n,))
    Cm_q = np.broadcast_to(Cm_q, (n,))
    Cl_delta = np.broadcast_to(Cl_delta, (n,))
    Cl_p = np.broadcast_to(Cl_p, (n,))
    fin_cant = np.broadcast_to(fin_cant, (n,))
    Mp = np.broadcast_to(M_pitch_external, (n,))
    My = np.broadcast_to(M_yaw_external, (n,))
    Mr = np.broadcast_to(M_roll_external, (n,))
    thrust_torque = np.broadcast_to(T_eq, (n,))
    Cl1 = np.broadcast_to(Cl1, (n,))
    cg_offset = np.broadcast_to(cg_offset, (n,))
    Gamma = np.broadcast_to(Gamma, (n,))
    CN_alpha = np.broadcast_to(CN_alpha, (n,))
    p = np.zeros(n)
    if roll_rate is not None:
        p = np.array(roll_rate, dtype=float)
    if roll_rate is None:
        p[0] = p0
    phi = np.zeros(n)
    phi[0] = phi0
    amplitudes = np.zeros((n, 2))
    amplitudes[0] = [R1_0, R2_0]
    growth = np.zeros((n, 2))
    frequency = np.zeros((n, 2))
    trim = np.zeros(n)
    moments = np.zeros(n)
    diagnostic = np.zeros(n, dtype=bool)
    for i in range(n):
        growth[i], frequency[i], trim[i], diagnostic[i] = transverse_modes(
            qbar[i], V[i], Ix[i], Iy[i], Iz[i], A, d, Cm_alpha[i],
            Cm_q[i], p[i], Mp[i], My[i],
        )
        moments[i] = roll_moment(
            qbar[i], V[i], p[i], A, d, fin_cant[i], Cl_delta[i], Cl_p[i],
            Cl1[i], N, phi[i], cg_offset[i], CN_alpha[i], trim[i],
            Gamma[i], thrust_torque[i], Mr[i],
        )
        if i < n - 1:
            dt = time[i + 1] - time[i]
            next_p, phi[i + 1], amplitudes[i + 1] = propagate_roll_modes(
                dt, p[i], phi[i], amplitudes[i], growth[i], moments[i], Ix[i])
            if roll_rate is None:
                p[i + 1] = next_p
    alpha = trim + np.abs(amplitudes).sum(axis=1)
    qalpha = qbar * alpha
    ia = np.argmax(alpha)
    iq = np.argmax(qalpha)
    return {
        "time": time.copy(), "p": p, "phi": phi,
        "T_eq": thrust_torque, "roll_moment": moments,
        "R1": amplitudes[:, 0], "R2": amplitudes[:, 1],
        "lambda_1": growth[:, 0], "lambda_2": growth[:, 1],
        "omega_1": frequency[:, 0], "omega_2": frequency[:, 1],
        "diagnostic_modes": diagnostic, "alpha_trim": trim,
        "alpha_upper": alpha, "q_alpha_upper": qalpha,
        "peak_alpha_rad": alpha[ia],
        "peak_alpha_deg": np.rad2deg(alpha[ia]),
        "peak_alpha_time": time[ia],
        "peak_q_alpha": qalpha[iq], "peak_q_alpha_time": time[iq],
    }


def aoa_from_flight_history(flight_history, geometry, Iz, **inputs):
    # Pull time, air loads, speed and inertias from each flight snapshot.
    time = []
    pressure = []
    speed = []
    roll_inertia = []
    pitch_inertia = []
    for row in flight_history:
        kin = row["kinematics"]
        time.append(kin.t)
        pressure.append(row["atmosphere"].q)
        speed.append(np.hypot(kin.vx, kin.vz))
        roll_inertia.append(row["mass_properties"]["Ixx"])
        pitch_inertia.append(row["mass_properties"]["Iyy"])
    return roll_pitch_yaw_upper_envelope(
        time=time, qbar=pressure, V=speed,
        Ix=roll_inertia, Iy=pitch_inertia, Iz=Iz,
        A=geometry.reference_area, d=geometry.diameter,
        fin_cant=np.deg2rad(geometry.fin_cant), **inputs,
    )
