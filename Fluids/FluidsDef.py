from __future__ import annotations

from typing import Any, Dict

import numpy as np


def coolprop_state(
    fluid: str,
    input_1: str,
    value_1: float,
    input_2: str,
    value_2: float,
) -> Dict[str, float]:
    """Evaluate a standard single-phase state with CoolProp."""

    from CoolProp.CoolProp import PropsSI

    def prop(name: str) -> float:
        return float(
            PropsSI(name, input_1, value_1, input_2, value_2, fluid)
        )

    return {
        "P": prop("P"),
        "T": prop("T"),
        "rho": prop("Dmass"),
        "h": prop("Hmass"),
        "u": prop("Umass"),
        "R": float(PropsSI("GAS_CONSTANT", fluid) / PropsSI("MOLAR_MASS", fluid)),
        "gamma": prop("Cpmass") / prop("Cvmass"),
        "mu": prop("VISCOSITY"),
        "k": prop("CONDUCTIVITY"),
        "cp": prop("Cpmass"),
        "beta": prop("ISOBARIC_EXPANSION_COEFFICIENT"),
    }


def compressible_mass_flux(
    P_upstream: float,
    P_downstream: float,
    T_upstream: float,
    gas_constant: float,
    gamma: float,
) -> float:
    """Ideal-gas isentropic mass flux, including choking."""

    pressure_ratio = np.clip(P_downstream / P_upstream, 0.0, 1.0)
    critical_ratio = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    if pressure_ratio <= critical_ratio:
        flow_function = np.sqrt(gamma) * (2.0 / (gamma + 1.0)) ** (
            (gamma + 1.0) / (2.0 * (gamma - 1.0))
        )
    else:
        flow_function = np.sqrt(
            2.0
            * gamma
            / (gamma - 1.0)
            * (
                pressure_ratio ** (2.0 / gamma)
                - pressure_ratio ** ((gamma + 1.0) / gamma)
            )
        )
    return float(P_upstream * flow_function / np.sqrt(gas_constant * T_upstream))


def isentropic_velocity(
    P_stagnation: float,
    P_exit: float,
    T_stagnation: float,
    gas_constant: float,
    gamma: float,
) -> float:
    """Ideal-gas velocity after isentropic expansion to the exit pressure."""

    if P_stagnation <= P_exit:
        return 0.0
    pressure_ratio = P_exit / P_stagnation
    return float(
        np.sqrt(
            2.0
            * gamma
            / (gamma - 1.0)
            * gas_constant
            * T_stagnation
            * (1.0 - pressure_ratio ** ((gamma - 1.0) / gamma))
        )
    )


def incompressible_mdot(CdA: float, density: float, pressure_drop: float) -> float:
    """Mass flow through an incompressible restriction."""

    return float(
        np.sign(pressure_drop)
        * CdA
        * np.sqrt(max(2.0 * density * abs(pressure_drop), 0.0))
    )


def incompressible_cda(
    mdot: float,
    density: float,
    pressure_drop: float,
) -> float:
    """Size an incompressible restriction for a target mass flow."""

    if density <= 0.0 or pressure_drop <= 0.0:
        raise ValueError("Incompressible sizing requires positive density and dP")
    return float(mdot / np.sqrt(2.0 * density * pressure_drop))


def compressible_cda(
    mdot: float,
    P_upstream: float,
    P_downstream: float,
    T_upstream: float,
    gas_constant: float,
    gamma: float,
) -> float:
    """Size a compressible restriction for a target mass flow."""

    mass_flux = compressible_mass_flux(
        P_upstream,
        P_downstream,
        T_upstream,
        gas_constant,
        gamma,
    )
    if mass_flux <= 0.0:
        raise ValueError("Compressible sizing requires positive mass flux")
    return float(mdot / mass_flux)


def combustion_properties(
    chamber_pressure: float,
    mixture_ratio: float,
    ambient_pressure: float,
    expansion_ratio: float,
    cea: Any,
    cstar_efficiency: float = 1.0,
    cf_efficiency: float = 1.0,
) -> Dict[str, float]:
    """Evaluate combustion properties with CEA."""

    if cea is None:
        raise ValueError("Combustion requires CEA")

    molecular_weight, gamma = cea.get_Chamber_MolWt_gamma(
        chamber_pressure,
        mixture_ratio,
        expansion_ratio,
    )
    return {
        "cstar": cea.get_Cstar(chamber_pressure, mixture_ratio)
        * cstar_efficiency,
        "Cf": cea.getFrozen_PambCf(
            ambient_pressure,
            chamber_pressure,
            mixture_ratio,
            expansion_ratio,
            1,
        )[1]
        * cf_efficiency,
        "R": 8314.462618 / molecular_weight,
        "gamma": gamma,
        "T": cea.get_Temperatures(
            chamber_pressure,
            mixture_ratio,
            expansion_ratio,
        )[0],
    }
