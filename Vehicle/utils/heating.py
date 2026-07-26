import numpy as np
from CoolProp.CoolProp import PropsSI

R = 287.05

def get_nose_heating() -> np.ndarray:
    return

def get_body_heating(x: np.ndarray, Tw: np.ndarray) -> np.ndarray:
    T2, p2, M2, v2 = get_post_shock()
    mu = get_mu(T2, p2)
    cp = get_cp(T2, p2)
    k = get_k(T2, p2)
    Pr = get_Pr(mu, cp, k)
    gamma = get_gamma(T2, p2)
    T_ref, mu_ref, rho_ref, Pr_ref = get_ref_props(T2, Tw, p2, M2, gamma, Pr)
    Re = get_Re(rho_ref, v2, x, mu_ref)
    Cf = get_Cf(Re)
    St = get_St(Cf, Pr_ref)
    Hr = 
    Hw = cp * Tw
    return rho_ref * v2 * St * (Hr - Hw)

def get_post_shock():
    return T2, p2, M2, v2

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