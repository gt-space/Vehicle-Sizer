from __future__ import annotations

from typing import Any, Dict

import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import root_scalar


class FluidsDef:
    """Reusable fluid, tank, and combustion-property equations."""

    @staticmethod
    def coolprop_state(
        fluid: str,
        input_1: str,
        value_1: float,
        input_2: str,
        value_2: float,
    ) -> Dict[str, float]:
        """Evaluate a standard single-phase state with CoolProp."""

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
        }

    @staticmethod
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

    @staticmethod
    def incompressible_mdot(CdA: float, density: float, pressure_drop: float) -> float:
        """Mass flow through an incompressible restriction."""

        return float(
            np.sign(pressure_drop)
            * CdA
            * np.sqrt(max(2.0 * density * abs(pressure_drop), 0.0))
        )

    @staticmethod
    def incompressible_cda(
        mdot: float,
        density: float,
        pressure_drop: float,
    ) -> float:
        """Size an incompressible restriction for a target mass flow."""

        if density <= 0.0 or pressure_drop <= 0.0:
            raise ValueError("Incompressible sizing requires positive density and dP")
        return float(mdot / np.sqrt(2.0 * density * pressure_drop))

    @classmethod
    def compressible_mdot(
        cls,
        CdA: float,
        P_upstream: float,
        P_downstream: float,
        T_upstream: float,
        gas_constant: float,
        gamma: float,
    ) -> float:
        """Mass flow through a compressible restriction."""

        return CdA * cls.compressible_mass_flux(
            P_upstream,
            P_downstream,
            T_upstream,
            gas_constant,
            gamma,
        )

    @classmethod
    def compressible_cda(
        cls,
        mdot: float,
        P_upstream: float,
        P_downstream: float,
        T_upstream: float,
        gas_constant: float,
        gamma: float,
    ) -> float:
        """Size a compressible restriction for a target mass flow."""

        mass_flux = cls.compressible_mass_flux(
            P_upstream,
            P_downstream,
            T_upstream,
            gas_constant,
            gamma,
        )
        if mass_flux <= 0.0:
            raise ValueError("Compressible sizing requires positive mass flux")
        return float(mdot / mass_flux)

    @classmethod
    def tank_compatibility(
        cls,
        m_liquid: float,
        U_liquid: float,
        m_gas: float,
        U_gas: float,
        tank_volume: float,
        liquid_fluid: str,
        gas_fluid: str,
        pressure_guess: float,
    ) -> Dict[str, Any]:
        """Solve liquid and ullage volumes at their common tank pressure."""

        u_gas = U_gas / m_gas
        u_liquid = U_liquid / m_liquid

        def volume_error(log_pressure: float) -> float:
            pressure = float(np.exp(log_pressure))
            rho_gas = cls.coolprop_state(
                gas_fluid, "P", pressure, "Umass", u_gas
            )["rho"]
            rho_liquid = cls.coolprop_state(
                liquid_fluid, "P", pressure, "Umass", u_liquid
            )["rho"]
            return m_gas / rho_gas + m_liquid / rho_liquid - tank_volume

        solution = root_scalar(
            volume_error,
            x0=np.log(pressure_guess),
            x1=np.log(pressure_guess * 1.001),
            method="secant",
        )
        if not solution.converged:
            raise RuntimeError("Tank pressure compatibility solve failed")

        pressure = float(np.exp(solution.root))
        gas = cls.coolprop_state(gas_fluid, "P", pressure, "Umass", u_gas)
        liquid = cls.coolprop_state(
            liquid_fluid, "P", pressure, "Umass", u_liquid
        )
        V_gas = m_gas / gas["rho"]
        V_liquid = m_liquid / liquid["rho"]
        return {
            "P": pressure,
            "gas": {**gas, "V": V_gas},
            "liquid": {**liquid, "V": V_liquid},
        }

    @staticmethod
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
            "h": cea.get_Chamber_H(
                chamber_pressure,
                mixture_ratio,
                expansion_ratio,
            ),
        }
