import numpy as np
from scipy.optimize import brentq
from CoolProp.CoolProp import PropsSI

GAMMA = 1.4
PR = 0.72
R_RECOV = np.cbrt(PR)  # turbulent recovery factor

def get_mu(T, p):
    return PropsSI("V", "T", T, "P", p, "air")

def get_Taw(Te: float, M: float) -> float:
    return Te * (1 + R_RECOV * (GAMMA - 1) / 2 * M**2)

def van_driest_II_Cf(Re_x: float, M: float, Te: float, Tw: float, p: float) -> float:
    """Mean turbulent skin friction, Van Driest II compressible transform of the
    Karman-Schoenherr law (Hopkins & Inouye, NASA TN D-6411, 1971)."""
    if M < 0.2:  # compressibility factors -> 0/0; use incompressible Karman-Schoenherr
        return brentq(lambda Cf: 1/np.sqrt(Cf) - (4.15*np.log10(Re_x*Cf) + 1.7), 1e-5, 0.05)
    A2 = R_RECOV * (GAMMA - 1) / 2 * M**2 * (Te / Tw)
    B = get_Taw(Te, M) / Tw - 1
    denom = np.sqrt(B**2 + 4*A2)
    F_theta = np.sqrt(A2) / (np.arcsin((2*A2 - B)/denom) + np.arcsin(B/denom))
    F_x = get_mu(Te, p) / get_mu(Tw, p)
    return brentq(lambda Cf: F_theta/np.sqrt(Cf) - (4.15*np.log10(Re_x*Cf*F_x) + 1.7), 1e-5, 0.05)

def _bridge(M: float, f_hi, M_lo: float = 0.9, M_hi: float = 1.2) -> float:
    """Transonic smoothstep: 0 below M_lo, f_hi(M) above M_hi, blended between."""
    if M <= M_lo:
        return 0.0
    if M >= M_hi:
        return f_hi(M)
    t = (M - M_lo) / (M_hi - M_lo)
    return t**2 * (3 - 2*t) * f_hi(M_hi)

def get_friction_drag(Cf: float, form_factor: float, Swet: float, Sref: float) -> float:
    return Cf * form_factor * Swet / Sref

def body_form_factor(L: float, D: float) -> float:
    """Hoerner slender-body form factor."""
    return 1 + 60 / (L / D)**3 + 0.0025 * (L / D)

def fin_form_factor(t_c: float) -> float:
    """Hoerner airfoil form factor."""
    return 1 + 2 * t_c + 60 * t_c**4

def get_nose_wave_drag(M: float, fineness: float) -> float:
    """Supersonic nose wave drag. Empirical pointed-nose fit (rocketry/RASAero lineage)."""
    return _bridge(M, lambda M: (1.586 + 1.834 / M**2) * np.arctan(0.5 / fineness)**1.69)

def get_base_drag(M: float, A_base: float, A_exit: float, Sref: float, power_on: bool) -> float:
    """Base drag, two-branch empirical (NACA base-pressure lineage). Plume fills exit area when powered."""
    A_eff = A_base - (A_exit if power_on else 0.0)
    cd = (0.12 + 0.13 * M**2) if M <= 1 else 0.25 / M
    return cd * A_eff / Sref

def get_boattail_drag(M: float, angle: float, A_boattail: float, Sref: float) -> float:
    """Boattail pressure drag. PLACEHOLDER: simplified slender-body scaling, Mach-independent above transonic."""
    return _bridge(M, lambda M: 2 * angle**2 * (A_boattail / Sref))

def get_fin_wave_drag(M: float, t_c: float, N: int, S_planform: float, Sref: float, K: float = 4/3) -> float:
    """Supersonic fin wave drag, linearized (Ackeret) thin-airfoil theory. K=4/3 for double-wedge."""
    return _bridge(M, lambda M: N * K * t_c**2 * (4 / np.sqrt(M**2 - 1)) * S_planform / Sref)

def get_CD(atm, geom: dict, power_on: bool = True, Tw: float = None) -> float:
    """Total drag coefficient (ref = frontal area) via DATCOM-style component build-up.

    geom keys: L, D, Swet_body, fineness, A_base, A_exit, boattail_angle, A_boattail,
               N_fins, Cr, t_c, S_fin_wet, S_fin_planform, Sref
    """
    M, Te, p = atm.Ma, atm.T, atm.p
    v = M * atm.a
    if Tw is None:
        Tw = get_Taw(Te, M)  # adiabatic wall if not coupled to heating.py
    mu_e = get_mu(Te, p)

    Cf_body = van_driest_II_Cf(v * geom["L"] / mu_e, M, Te, Tw, p)
    Cf_fin = van_driest_II_Cf(v * geom["Cr"] / mu_e, M, Te, Tw, p)

    return (
        get_friction_drag(Cf_body, body_form_factor(geom["L"], geom["D"]), geom["Swet_body"], geom["Sref"])
        + get_friction_drag(Cf_fin, fin_form_factor(geom["t_c"]), geom["S_fin_wet"], geom["Sref"])
        + get_nose_wave_drag(M, geom["fineness"])
        + get_base_drag(M, geom["A_base"], geom["A_exit"], geom["Sref"], power_on)
        + get_boattail_drag(M, geom["boattail_angle"], geom["A_boattail"], geom["Sref"])
        + get_fin_wave_drag(M, geom["t_c"], geom["N_fins"], geom["S_fin_planform"], geom["Sref"])
    )

def get_drag_force(atm, geom: dict, power_on: bool = True, Tw: float = None) -> float:
    return atm.q * geom["Sref"] * get_CD(atm, geom, power_on, Tw)
