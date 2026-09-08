import numpy as np
from CoolProp.CoolProp import PropsSI

R = 287.05
MANGLER_TURB = 3**0.2

def get_nose_stag(atm, R_n: float) -> float:
    K = 1.7415e-4
    v = atm.Ma * atm.a
    return K * np.sqrt(atm.rho / R_n) * v**3

def get_nose_heating(atm, R_n: float, radius: np.ndarray, dx: float, Tw: np.ndarray) -> np.ndarray:
    dr = np.diff(radius, prepend=radius[0])
    s = np.cumsum(np.sqrt(dx**2 + dr**2))

    q_stag = get_nose_stag(atm, R_n)
    theta = s / R_n
    q_cap = q_stag * np.cos(np.minimum(theta, np.pi / 2))

    delta = np.arctan(radius[-1] / s[-1])
    T2, p2, M2, v2 = get_post_shock(atm, delta)
    mu = get_mu(T2, p2)
    cp = get_cp(T2, p2)
    k = get_k(T2, p2)
    Pr = get_Pr(mu, cp, k)
    gamma = get_gamma(T2, p2)
    T_ref, mu_ref, rho_ref, Pr_ref = get_ref_props(T2, Tw, p2, M2, gamma, Pr)
    Re = get_Re(rho_ref, v2, s, mu_ref)
    Hr = cp * T2 * (1 + get_recov_factor(Pr) * (0.5 * (gamma - 1)) * M2**2)
    Hw = cp * Tw
    St = MANGLER_TURB * get_St(get_Cf(Re), Pr_ref)
    q_flank = rho_ref * v2 * St * (Hr - Hw)

    s_tan = R_n * (np.pi / 2 - delta)
    return np.where(s <= s_tan, q_cap, q_flank)

def get_body_heating(x: np.ndarray, Tw: np.ndarray, atm, theta: float) -> np.ndarray:
    T2, p2, M2, v2 = get_post_shock(atm, theta)
    mu = get_mu(T2, p2)
    cp = get_cp(T2, p2)
    k = get_k(T2, p2)
    Pr = get_Pr(mu, cp, k)
    gamma = get_gamma(T2, p2)
    T_ref, mu_ref, rho_ref, Pr_ref = get_ref_props(T2, Tw, p2, M2, gamma, Pr)
    Re = get_Re(rho_ref, v2, x, mu_ref)
    Cf = get_Cf(Re)
    St = get_St(Cf, Pr_ref)
    Hr = cp * T2 * (1 + get_recov_factor(Pr) * (0.5 * (gamma - 1)) * M2**2)
    Hw = cp * Tw
    return rho_ref * v2 * St * (Hr - Hw)

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
    mu = np.arcsin(1 / M)
    beta = np.linspace(mu, np.pi / 2, 500)
    num = M**2 * np.sin(beta)**2 - 1
    den = M**2 * (gamma + np.cos(2 * beta)) + 2
    theta_b = np.arctan(2 / np.tan(beta) * num / den)
    i = np.argmax(theta_b)
    return np.interp(theta, theta_b[:i + 1], beta[:i + 1])

def get_ref_props(T, Tw, p, M, gamma, Pr):
    T_ref = get_ref_temp(T, Tw, M, gamma, Pr)
    mu_ref = get_ref_visc(T_ref)
    rho_ref = get_ref_density(p, R, T_ref)
    cp_ref = get_cp(T, p)
    k_ref = get_k(T, p)
    Pr_ref = get_Pr(mu_ref, cp_ref, k_ref)
    return T_ref, mu_ref, rho_ref, Pr_ref

def get_ref_temp(T: float, Tw: np.ndarray, M: float, gamma: float, Pr: np.ndarray) -> np.ndarray:
    r = get_recov_factor(Pr)
    return T * (0.5 * (1 + Tw / T) + 0.16 * r * (0.5 * (gamma - 1)) * M**2)

def get_ref_visc(T: np.ndarray) -> np.ndarray:
    return 1.716e-5 * (T / 273.15)**1.5 * ((273.15 + 110.4) / (T + 110.4))

def get_ref_density(p, R, T):
    return p / (R * T)

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
