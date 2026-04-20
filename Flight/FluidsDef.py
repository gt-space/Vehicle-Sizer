from __future__ import annotations

class FluidsDef:

    @staticmethod
    def orifice_capacity_choked(*args, **kwargs) -> float:
        # TODO: 
        return 0.0

    @staticmethod
    def orifice_capacity_unchoked(*args, **kwargs) -> float:
        # TODO: 
        return 0.0
    
    def _tank_compatability(m_liq, U_liq, m_gas, U_gas, V_tank, tank_obj):
        """
        Solve for ullage and liquid volumes from (m, U) states.

        Returns:
            V_liq, V_gas,
            P, (common tank pressure)
            liquid_state, gas_state
        """

        eps = 1e-6
        Vg_min = eps
        Vg_max = V_tank - eps

        def residual(Vg):
            Vl = V_tank - Vg

            rho_g = m_gas / Vg
            u_g = U_gas / m_gas

            Pg = tank_obj.gas_eos_rho_u(rho_g, u_g)

            rho_l = m_liq / Vl
            u_l = U_liq / m_liq

            Pl = tank_obj.liq_eos_rho_u(rho_l, u_l)

            return Pl - Pg  

        sol = root_scalar(residual, bracket=[Vg_min, Vg_max], method="bisect")

        if not sol.converged:
            raise RuntimeError("Tank volume solve failed")

        Vg = sol.root
        Vl = V_tank - Vg

        rho_g = m_gas / Vg
        u_g = U_gas / m_gas
        Pg, Tg, hg = tank_obj.gas_eos_rho_u(rho_g, u_g)

        rho_l = m_liq / Vl
        u_l = U_liq / m_liq
        Pl, Tl, hl = tank_obj.liq_eos_rho_u(rho_l, u_l)

        return {
            "V_gas": Vg,
            "V_liq": Vl,
            "P": Pg,  
            "gas": {
                "rho": rho_g,
                "T": Tg,
                "h": hg,
                "u": u_g,
            },
            "liquid": {
                "rho": rho_l,
                "T": Tl,
                "h": hl,
                "u": u_l,
            },
        }