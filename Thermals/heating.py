import warnings

import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq, minimize_scalar

R = 287.05
MANGLER_TURB = 3**0.2

def get_nose_stag(atm, R_n: float) -> float:
    K = 1.7415e-4
    v = atm.Ma * atm.a
    return K * np.sqrt(atm.rho / R_n) * v**3

def get_nose_heating(atm, R_n: float, radius: np.ndarray, dx: float, Tw: np.ndarray) -> np.ndarray:
    check_continuum(atm, R_n)
    if abs(atm.Ma) < 1.0e-8:
        return np.zeros_like(Tw, dtype=float)
    dr = np.diff(radius, prepend=radius[0])
    s = np.cumsum(np.sqrt(dx**2 + dr**2))

    if abs(atm.Ma) <= 1.0:
        return _boundary_layer_heating(atm.T, atm.p, abs(atm.Ma), abs(atm.Ma * atm.a), s, Tw)

    q_stag = get_nose_stag(atm, R_n)
    theta = s / R_n
    q_cap = q_stag * np.cos(np.minimum(theta, np.pi / 2))

    delta = np.arctan(radius[-1] / s[-1])
    T2, p2, M2, v2 = get_surface_flow(atm, delta)
    q_flank = MANGLER_TURB * _boundary_layer_heating(T2, p2, M2, v2, s, Tw)

    s_tan = R_n * (np.pi / 2 - delta)
    return np.where(s <= s_tan, q_cap, q_flank)

def get_body_heating(x: np.ndarray, Tw: np.ndarray, atm, theta: float) -> np.ndarray:
    check_continuum(atm, x)
    if abs(atm.Ma) < 1.0e-8:
        return np.zeros_like(Tw, dtype=float)
    T2, p2, M2, v2 = get_surface_flow(atm, theta)
    return _boundary_layer_heating(T2, p2, M2, v2, x, Tw)


def _boundary_layer_heating(T, p, M, velocity, x, Tw):
    mu = get_mu(T, p)
    cp = get_cp(T, p)
    k = get_k(T, p)
    Pr = get_Pr(mu, cp, k)
    gamma = get_gamma(T, p)
    T_ref, mu_ref, rho_ref, Pr_ref = get_ref_props(T, Tw, p, M, gamma, Pr)
    cp_ref = get_cp(T_ref, p)
    Re = get_Re(rho_ref, velocity, x, mu_ref)
    Cf = get_Cf(Re)
    St = get_St(Cf, Pr_ref)
    Hr = cp_ref * T * (1 + get_recov_factor(Pr) * (0.5 * (gamma - 1)) * M**2)
    Hw = cp_ref * Tw
    return rho_ref * velocity * St * (Hr - Hw)


def get_surface_flow(atm, theta: float):
    """Return freestream, attached-shock, or detached normal-shock conditions."""

    if abs(atm.Ma) <= 1.0:
        return atm.T, atm.p, abs(atm.Ma), abs(atm.Ma * atm.a)
    try:
        return get_post_shock(atm, theta)
    except ValueError as error:
        if "No attached oblique shock" not in str(error):
            raise
        return get_normal_shock(atm)


def get_normal_shock(atm):
    gamma = get_gamma(atm.T, atm.p)
    M1 = abs(atm.Ma)
    p_ratio = 1 + 2 * gamma / (gamma + 1) * (M1**2 - 1)
    rho_ratio = ((gamma + 1) * M1**2) / ((gamma - 1) * M1**2 + 2)
    T2 = atm.T * p_ratio / rho_ratio
    p2 = atm.p * p_ratio
    M2 = np.sqrt((1 + 0.5 * (gamma - 1) * M1**2) / (gamma * M1**2 - 0.5 * (gamma - 1)))
    return T2, p2, M2, M2 * np.sqrt(gamma * R * T2)


def get_recovery_temperature(atm, theta: float = 0.0):
    T, p, M, _ = get_surface_flow(atm, theta)
    mu = get_mu(T, p)
    cp = get_cp(T, p)
    Pr = get_Pr(mu, cp, get_k(T, p))
    gamma = get_gamma(T, p)
    return T * (1 + get_recov_factor(Pr) * 0.5 * (gamma - 1) * M**2)

def get_post_shock(atm, theta: float):
    gamma = get_gamma(atm.T, atm.p)
    M1 = atm.Ma
    beta = get_oblique_beta(M1, theta, gamma)
    Mn1 = M1 * np.sin(beta)
    p_ratio = 1 + 2 * gamma / (gamma + 1) * (Mn1**2 - 1)
    p2 = atm.p * p_ratio
    T2 = atm.T * p_ratio * ((gamma - 1) * Mn1**2 + 2) / ((gamma + 1) * Mn1**2)
    Mn2 = np.sqrt((1 + 0.5 * (gamma - 1) * Mn1**2) / (gamma * Mn1**2 - 0.5 * (gamma - 1)))
    M2 = Mn2 / np.sin(beta - theta)
    v2 = M2 * np.sqrt(gamma * R * T2)
    return T2, p2, M2, v2

def get_oblique_beta(M: float, theta: float, gamma: float) -> float:
    if not np.isfinite(M) or M <= 1.0:
        raise ValueError("An attached oblique shock requires Mach > 1")
    if not np.isfinite(theta) or not 0.0 <= theta < np.pi / 2:
        raise ValueError("Shock deflection angle must be in [0, pi/2)")
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("Specific-heat ratio must be greater than one")

    mu = np.arcsin(1 / M)
    if theta == 0.0:
        return mu

    beta_peak = minimize_scalar(
        lambda beta: -get_shock_deflection(M, beta, gamma),
        bounds=(mu, np.pi / 2),
        method="bounded",
    ).x
    theta_max = get_shock_deflection(M, beta_peak, gamma)
    if theta > theta_max:
        raise ValueError(
            f"No attached oblique shock for theta={np.degrees(theta):.3g} deg "
            f"at Mach {M:.3g}; maximum is {np.degrees(theta_max):.3g} deg"
        )
    if np.isclose(theta, theta_max):
        return beta_peak
    return brentq(
        lambda beta: get_shock_deflection(M, beta, gamma) - theta,
        mu,
        beta_peak,
    )


def get_shock_deflection(M: float, beta: float, gamma: float) -> float:
    num = M**2 * np.sin(beta)**2 - 1
    den = M**2 * (gamma + np.cos(2 * beta)) + 2
    return np.arctan(2 / np.tan(beta) * num / den)

def get_ref_props(T, Tw, p, M, gamma, Pr):
    T_ref = get_ref_temp(T, Tw, M, gamma, Pr)
    mu_ref = get_ref_visc(T_ref)
    rho_ref = get_ref_density(p, R, T_ref)
    cp_ref = get_cp(T_ref, p)
    k_ref = get_k(T_ref, p)
    Pr_ref = get_Pr(mu_ref, cp_ref, k_ref)
    return T_ref, mu_ref, rho_ref, Pr_ref

def get_ref_temp(T: float, Tw: np.ndarray, M: float, gamma: float, Pr: np.ndarray) -> np.ndarray:
    r = get_recov_factor(Pr)
    return T * (0.5 * (1 + Tw / T) + 0.16 * r * (0.5 * (gamma - 1)) * M**2)

def get_ref_visc(T: np.ndarray) -> np.ndarray:
    return 1.716e-5 * (T / 273.15)**1.5 * ((273.15 + 110.4) / (T + 110.4))

def get_ref_density(p, R, T):
    return p / (R * T)


def get_Kn(atm, length):
    length = np.asarray(length, dtype=float)
    if np.any(~np.isfinite(length)) or np.any(length <= 0.0):
        raise ValueError("Knudsen characteristic length must be positive")
    mean_free_path = atm.mu / atm.p * np.sqrt(np.pi * R * atm.T / 2.0)
    return mean_free_path / length


def check_continuum(atm, length, limit: float = 0.01):
    Kn = get_Kn(atm, length)
    if np.any(Kn >= limit):
        warnings.warn(
            "Continuum heating correlation used at Kn >= 0.01",
            RuntimeWarning,
            stacklevel=2,
        )
    return Kn

def get_recov_factor(Pr: np.ndarray) -> np.ndarray:
    return np.cbrt(Pr)

def get_mu(T, p):
    return PropsSI("V", "T", T, "P", p, "air")

def get_cp(T, p):
    return PropsSI("CPMASS", "T", T, "P", p, "air")

def get_k(T, p):
    return PropsSI("L", "T", T, "P", p, "air")

def get_gamma(T, p):
    return get_cp(T, p) / PropsSI("CVMASS", "T", T, "P", p, "air")

def get_Re(rho: float, v: float, x: np.ndarray, mu: float) -> np.ndarray:
    return (rho * v * x) / mu

def get_Cf(Re: np.ndarray) -> np.ndarray:
    return 0.02296 / Re**0.139

def get_St(Cf: np.ndarray, Pr: np.ndarray) -> np.ndarray:
    return 0.5 * Cf * Pr**(-2/3)

def get_Pr(mu: np.ndarray, cp, k) -> np.ndarray:
    return (mu * cp) / k
