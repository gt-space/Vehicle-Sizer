from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
from scipy.optimize import root_scalar
from .types import KinematicsState, AtmosState, ThermalOut, AeroOut, FluidOut
from .FluidNetwork import FluidNetwork

class PropSystem:

    def __init__(self, cfg: Dict[str, Any]):

        self.cfg = cfg
        self.model = cfg["press_model"]

        self._init_model(cfg)
        self.target_ladder = self._build_pressure_ladder(cfg)

        nodes, branches = self._wire_network(cfg)
        nodes, branches = self._size_network(nodes, branches, cfg)

        self.network = FluidNetwork(nodes=nodes, branches=branches)

    def _wire_network(self,
                      cfg: Dict[str, Any],
                      ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:

        if self.model == "pump_fed":
            return self._template_pump_fed(cfg)
        elif self.model == "pressure_fed":
            return self._template_pressure_fed(cfg)
        elif self.model == "blowdown":
            return self._template_blowdown(cfg)
        else:
            raise ValueError(f"Unknown press_model={self.model}")

    def _size_network(self,
                      nodes: Dict[str, Dict[str, Any]],
                      branches: Dict[str, Dict[str, Any]],
                      cfg: Dict[str, Any],
                      ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
        self._size_nodes(nodes, cfg)
        self._size_branches(branches, cfg)
        return nodes, branches
    
    def _template_pump_fed(self,
                           cfg: Dict[str, Any]
                           ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:

        nodes = {"press_tank": {"type": "gas_volume","V": None,"state0": {"P": None, "T": None},"steady": False},
                 "ox_ullage": {"type": "gas_volume","V": None,"state0": {"P": None, "T": None},"steady": True},
                 "fuel_ullage": {"type": "gas_volume","V": None,"state0": {"P": None, "T": None},"steady": True,},
                 "ox_pump_in": {"type": "liquid_volume", "steady": True},
                 "ox_pump_out": {"type": "liquid_volume", "steady": True},
                 "ox_inj_in": {"type": "liquid_volume", "steady": True},
                 "fuel_pump_in": {"type": "liquid_volume", "steady": True},
                 "fuel_pump_out": {"type": "liquid_volume", "steady": True},
                 "fuel_inj_in": {"type": "liquid_volume", "steady": True},
                 "thrust_chamber": {"type": "comb_device", "steady": True},
                 "ambient": {"type": "boundary_pressure", "steady": True}}

        branches = {"OX_BANGBANG": {"type": "gas_orifice","fluid": "gas","from": "press_tank","to": "ox_ullage","CdA": None,"steady": False},
                    "FUEL_BANGBANG": {"type": "gas_orifice","fluid": "gas","from": "press_tank","to": "fuel_ullage","CdA": None,"steady": False},
                    "OX_TANK_PUMP": {"type": "liquid_loss","fluid": "ox","from": "ox_ullage","to": "ox_pump_in","CdA": None,"steady": True},
                    "OX_PUMP": {"type": "pump","fluid": "ox","from": "ox_pump_in","to": "ox_pump_out","dP": self.ox_Pump_head,"CdA": None,"steady": True},
                    "OX_PUMP_INJ": {"type": "liquid_loss","fluid": "ox","from": "ox_pump_out","to": "ox_inj_in","CdA": None,"steady": True},
                    "OX_INJ": {"type": "liquid_orifice","fluid": "ox","from": "ox_inj_in","to": "thrust_chamber","CdA": None,"steady": True},
                    "FUEL_TANK_PUMP": {"type": "liquid_loss","fluid": "fuel","from": "fuel_ullage","to": "fuel_pump_in","CdA": None,"steady": True},
                    "FUEL_PUMP": {"type": "pump","fluid": "fuel","from": "fuel_pump_in","to": "fuel_pump_out","dP": self.ox_Pump_head, "CdA": None,"steady": True},
                    "FUEL_PUMP_INJ": {"type": "liquid_loss","fluid": "fuel","from": "fuel_pump_out","to": "fuel_inj_in","CdA": None,"steady": True},
                    "FUEL_INJ": {"type": "liquid_orifice","fluid": "fuel","from": "fuel_inj_in","to": "thrust_chamber","CdA": None,"steady": True},
                    "NOZZLE": {"type": "gas_orifice","fluid": "gas","from": "thrust_chamber","to": "ambient","CdA": None,"steady": True}}

        return nodes, branches

    def _template_pressure_fed(self,
                               cfg: Dict[str, Any],
                               ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:

        nodes = {"press_tank": {"type": "gas_volume","V": None,"state0": {"P": None, "T": None},"steady": False},
                 "ox_ullage": {"type": "gas_volume","V": None,"state0": {"P": None, "T": None},"steady": True},
                 "fuel_ullage": {"type": "gas_volume","V": None,"state0": {"P": None, "T": None},"steady": True},
                 "ox_inj_in": {"type": "liquid_volume", "steady": True},
                 "fuel_inj_in": {"type": "liquid_volume", "steady": True},
                 "thrust_chamber": {"type": "comb_device", "steady": True},
                 "ambient": {"type": "boundary_pressure", "steady": True}}

        branches = {"OX_BANGBANG": {"type": "gas_orifice","fluid": "gas","from": "press_tank","to": "ox_ullage","CdA": None,"steady": False},
                    "FUEL_BANGBANG": {"type": "gas_orifice","fluid": "gas","from": "press_tank","to": "fuel_ullage","CdA": None,"steady": False},
                    "OX_TANK_INJ": {"type": "liquid_loss","fluid": "ox","from": "ox_ullage","to": "ox_inj_in","CdA": None,"steady": True},
                    "OX_INJ": {"type": "liquid_orifice","fluid": "ox","from": "ox_inj_in","to": "thrust_chamber","CdA": None,"steady": True},
                    "FUEL_TANK_INJ": {"type": "liquid_loss","fluid": "fuel","from": "fuel_ullage","to": "fuel_inj_in","CdA": None,"steady": True},
                    "FUEL_INJ": {"type": "liquid_orifice","fluid": "fuel","from": "fuel_inj_in","to": "thrust_chamber","CdA": None,"steady": True},
                    "NOZZLE": {"type": "gas_orifice","fluid": "gas","from": "thrust_chamber","to": "ambient","CdA": None,"steady": True}}

        return nodes, branches

    def _template_blowdown(self,
                           cfg: Dict[str, Any],
                           ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:

        nodes = {"ox_ullage": {"type": "gas_volume","V": None,"state0": {"P": None, "T": None},"steady": True},
                 "fuel_ullage": {"type": "gas_volume","V": None,"state0": {"P": None, "T": None},"steady": True},
                 "ox_inj_in": {"type": "liquid_volume", "steady": True},
                 "fuel_inj_in": {"type": "liquid_volume", "steady": True},
                 "thrust_chamber": {"type": "comb_device", "steady": True},
                 "ambient": {"type": "boundary_pressure", "steady": True}}

        branches = {"OX_TANK_INJ": {"type": "liquid_loss","fluid": "ox","from": "ox_ullage","to": "ox_inj_in","CdA": None,"steady": True},
                    "OX_INJ": {"type": "liquid_orifice","fluid": "ox","from": "ox_inj_in","to": "thrust_chamber","CdA": None,"steady": True},
                    "FUEL_TANK_INJ": {"type": "liquid_loss","fluid": "fuel","from": "fuel_ullage","to": "fuel_inj_in","CdA": None,"steady": True},
                    "FUEL_INJ": {"type": "liquid_orifice","fluid": "fuel","from": "fuel_inj_in","to": "thrust_chamber","CdA": None,"steady": True},
                    "NOZZLE": {"type": "gas_orifice","fluid": "gas","from": "thrust_chamber","to": "ambient","CdA": None,"steady": True}}

        return nodes, branches

    def _init_model(self, cfg: Dict[str, Any]) -> None:
        # initialize from config based on model
        self.duty_cyl = cfg['duty_cyl']
        #rewrite to reduce rewritten lines
        if self.model == "pump_fed":
            self.Pc_target = cfg['Pc_target']
            self.fuel_Pump_head = cfg["fuel_Pump_head"]
            self.ox_Pump_head = cfg["ox_pump_head"]
            self.fuel_inj_stiffness = cfg["fuel_inj_stiffness"]
            self.ox_inj_stiffness = cfg["ox_inj_stiffness"]
            self.fuel_inj_pumpout_dp = cfg["fuel_inj_pumpout_dp"]
            self.ox_inj_pumpout_dp = cfg["ox_inj_pumpout_dp"]
            self.fuel_pumpin_tank_dp = cfg["fuel_pumpin_tank_dp"]
            self.ox_pumpin_tank_dp = cfg["ox_pumpin_tank_dp"]
            self.ox_pressurant_dp = cfg["ox_pressurant_dp"]
            self.fuel_pressurant_dp = cfg["fuel_pressurant_dp"]
        elif self.model == "pressure_fed":
            self.Pc_target = cfg['Pc_target']
            self.fuel_inj_stiffness = cfg["fuel_inj_stiffness"]
            self.ox_inj_stiffness = cfg["ox_inj_stiffness"]
            self.fuel_tank_inj_dp = cfg["fuel_tank_inj_dp"]
            self.ox_tank_inj_dp = cfg["ox_tank_inj_dp"]
            self.ox_pressurant_dp = cfg["ox_pressurant_dp"]
            self.fuel_pressurant_dp = cfg["fuel_pressurant_dp"]
        elif self.model == "blowdown":
            self.Pc_target = cfg['Pc_target']
            self.fuel_inj_stiffness = cfg["fuel_inj_stiffness"]
            self.ox_inj_stiffness = cfg["ox_inj_stiffness"]
            self.fuel_tank_inj_dp = cfg["fuel_tank_inj_dp"]
            self.ox_tank_inj_dp = cfg["ox_tank_inj_dp"]
        else:
            raise ValueError(f"Unknown press_model={self.model}")

    def _build_pressure_ladder(self, cfg: Dict[str, Any]) -> Dict[str, float]:
        # build pressure ladder based on config inputs, copv pressure calculated separately
        if self.model == "pump_fed":
            Pc = self.Pc_target
            Pox_inj = Pc + (Pc * self.ox_inj_stiffness)
            Pox_pump_outlet = Pox_inj + self.ox_inj_pumpout_dp
            Pox_pump_inlet = Pox_pump_outlet - self.ox_Pump_head
            Pox_tank = Pox_pump_inlet + self.ox_pumpin_tank_dp
            Pfuel_inj = Pc + (Pc * self.fuel_inj_stiffness)
            Pfuel_pump_outlet = Pfuel_inj + self.fuel_inj_pumpout_dp
            Pfuel_pump_inlet = Pfuel_pump_outlet - self.fuel_Pump_head
            Pfuel_tank = Pfuel_pump_inlet + self.fuel_pumpin_tank_dp

            return {
                "Pc_target": Pc,
                "Pox_inj": Pox_inj,
                "Pox_pump_outlet": Pox_pump_outlet,
                "Pox_pump_inlet": Pox_pump_inlet,
                "Pox_tank": Pox_tank,
                "Pfuel_inj": Pfuel_inj,
                "Pfuel_pump_outlet": Pfuel_pump_outlet,
                "Pfuel_pump_inlet": Pfuel_pump_inlet,
                "Pfuel_tank": Pfuel_tank,
            }
        if self.model in ("pressure_fed" , "blowdown"):
            Pc = self.Pc_target
            Pox_inj = Pc + (Pc * self.ox_inj_stiffness)
            Pox_tank = Pox_inj + self.ox_tank_inj_dp
            Pfuel_inj = Pc + (Pc * self.fuel_inj_stiffness)
            Pfuel_tank = Pfuel_inj + self.fuel_tank_inj_dp

            return {
                "Pc_target": Pc,
                "Pox_inj": Pox_inj,
                "Pox_tank": Pox_tank,
                "Pfuel_inj": Pfuel_inj,
                "Pfuel_tank": Pfuel_tank,
            }

    def _size_branches(self, branches: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> None:
        pass

    def _size_nodes(self, nodes: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> None:
        pass

    def _has_press_tank(self) -> bool:
        return self.model in ("pump_fed", "pressure_fed")

    def _pack_state(self) -> np.ndarray:
        x = []

        if self._has_press_tank():
            x.extend([
                self.copv.m_gas,
                self.copv.U_gas,
            ])

        x.extend([
            self.ox_tank.m_liq,
            self.ox_tank.U_liq,
            self.ox_tank.m_ull,
            self.ox_tank.U_ull,

            self.fuel_tank.m_liq,
            self.fuel_tank.U_liq,
            self.fuel_tank.m_ull,
            self.fuel_tank.U_ull,
        ])

        return np.array(x, dtype=float)

    def _unpack_state(self, x: np.ndarray) -> Dict[str, float]:
        i = 0
        s: Dict[str, float] = {}

        if self._has_press_tank():
            s["copv_m"] = float(x[i]); i += 1
            s["copv_U"] = float(x[i]); i += 1
        else:
            s["copv_m"] = None
            s["copv_U"] = None

        s["ox_liq_m"] = float(x[i]); i += 1
        s["ox_liq_U"] = float(x[i]); i += 1
        s["ox_ull_m"] = float(x[i]); i += 1
        s["ox_ull_U"] = float(x[i]); i += 1

        s["fuel_liq_m"] = float(x[i]); i += 1
        s["fuel_liq_U"] = float(x[i]); i += 1
        s["fuel_ull_m"] = float(x[i]); i += 1
        s["fuel_ull_U"] = float(x[i]); i += 1

        return s


    def _commit_state(self, x: np.ndarray) -> None:
        s = self._unpack_state(x)

        if self._has_press_tank():
            self.copv.m_gas = s["copv_m"]
            self.copv.U_gas = s["copv_U"]

        self.ox_tank.m_liq = s["ox_liq_m"]
        self.ox_tank.U_liq = s["ox_liq_U"]
        self.ox_tank.m_ull = s["ox_ull_m"]
        self.ox_tank.U_ull = s["ox_ull_U"]

        self.fuel_tank.m_liq = s["fuel_liq_m"]
        self.fuel_tank.U_liq = s["fuel_liq_U"]
        self.fuel_tank.m_ull = s["fuel_ull_m"]
        self.fuel_tank.U_ull = s["fuel_ull_U"]

    def residual(self, x_guess: np.ndarray, x_prev: np.ndarray, dt: float, Qdot: Dict[str, float], atm):

        trial = self._unpack_state(x_guess)
        prev = self._unpack_state(x_prev)

        if self._has_press_tank():
            Pcopv, Tcopv, rho_copv, h_copv, u_copv_spec = self.copv.get_gas_state_m_U(
                m_gas=trial["copv_m"],
                U_total=trial["copv_U"],
            )
            Pcopv_prev, Tcopv_prev, rho_copv_prev, h_copv_prev, u_copv_prev_spec = self.copv.get_gas_state_m_U(
                m_gas=prev["copv_m"],
                U_total=prev["copv_U"],
            )
        else:
            Pcopv = 0.0
            Tcopv = 0.0
            rho_copv = 0.0
            h_copv = 0.0
            u_copv_spec = 0.0

        ox_state = self._tank_compatability(
            m_liq=trial["ox_liq_m"],
            U_liq=trial["ox_liq_U"],
            m_gas=trial["ox_ull_m"],
            U_gas=trial["ox_ull_U"],
            V_tank=self.ox_tank.volume,
            tank_obj=self.ox_tank,
        )

        ox_state_prev = self._tank_compatability(
            m_liq=prev["ox_liq_m"],
            U_liq=prev["ox_liq_U"],
            m_gas=prev["ox_ull_m"],
            U_gas=prev["ox_ull_U"],
            V_tank=self.ox_tank.volume,
            tank_obj=self.ox_tank,
        )

        fuel_state = self._tank_compatability(
            m_liq=trial["fuel_liq_m"],
            U_liq=trial["fuel_liq_U"],
            m_gas=trial["fuel_ull_m"],
            U_gas=trial["fuel_ull_U"],
            V_tank=self.fuel_tank.volume,
            tank_obj=self.fuel_tank,
        )

        fuel_state_prev = self._tank_compatability(
            m_liq=prev["fuel_liq_m"],
            U_liq=prev["fuel_liq_U"],
            m_gas=prev["fuel_ull_m"],
            U_gas=prev["fuel_ull_U"],
            V_tank=self.fuel_tank.volume,
            tank_obj=self.fuel_tank,
        )

        V_ox_liq = ox_state["V_liq"]
        V_ox_ull = ox_state["V_gas"]
        Pox = ox_state["P"]

        rho_ox_liq = ox_state["liquid"]["rho"]
        T_ox_liq = ox_state["liquid"]["T"]
        h_ox_liq = ox_state["liquid"]["h"]
        u_ox_liq_spec = ox_state["liquid"]["u"]

        rho_ox_ull = ox_state["gas"]["rho"]
        Tox = ox_state["gas"]["T"]
        h_ox_gas = ox_state["gas"]["h"]
        u_ox_gas_spec = ox_state["gas"]["u"]

        V_ox_liq_prev = ox_state_prev["V_liq"]
        V_ox_ull_prev = ox_state_prev["V_gas"]
        Pox_prev = ox_state_prev["P"]

        rho_ox_liq_prev = ox_state_prev["liquid"]["rho"]
        T_ox_liq_prev = ox_state_prev["liquid"]["T"]
        h_ox_liq_prev = ox_state_prev["liquid"]["h"]
        u_ox_liq_prev_spec = ox_state_prev["liquid"]["u"]

        rho_ox_ull_prev = ox_state_prev["gas"]["rho"]
        Tox_prev = ox_state_prev["gas"]["T"]
        h_ox_gas_prev = ox_state_prev["gas"]["h"]
        u_ox_gas_prev_spec = ox_state_prev["gas"]["u"]

        V_fuel_liq = fuel_state["V_liq"]
        V_fuel_ull = fuel_state["V_gas"]
        Pfuel = fuel_state["P"]

        rho_fuel_liq = fuel_state["liquid"]["rho"]
        T_fuel_liq = fuel_state["liquid"]["T"]
        h_fuel_liq = fuel_state["liquid"]["h"]
        u_fuel_liq_spec = fuel_state["liquid"]["u"]

        rho_fuel_ull = fuel_state["gas"]["rho"]
        Tfuel = fuel_state["gas"]["T"]
        h_fuel_gas = fuel_state["gas"]["h"]
        u_fuel_gas_spec = fuel_state["gas"]["u"]

        V_fuel_liq_prev = fuel_state_prev["V_liq"]
        V_fuel_ull_prev = fuel_state_prev["V_gas"] 
        Pfuel_prev = fuel_state_prev["P"]

        rho_fuel_liq_prev = fuel_state_prev["liquid"]["rho"]
        T_fuel_liq_prev = fuel_state_prev["liquid"]["T"]
        h_fuel_liq_prev = fuel_state_prev["liquid"]["h"]
        u_fuel_liq_prev_spec = fuel_state_prev["liquid"]["u"]

        rho_fuel_ull_prev = fuel_state_prev["gas"]["rho"]
        Tfuel_prev = fuel_state_prev["gas"]["T"]
        h_fuel_gas_prev = fuel_state_prev["gas"]["h"]
        u_fuel_gas_prev_spec = fuel_state_prev["gas"]["u"]

        bcs = {
            "ox_ullage": {"P": Pox, "T": Tox},
            "fuel_ullage": {"P": Pfuel, "T": Tfuel},
            "ambient": {"P": atm.P},
        }

        if self._has_press_tank():
            bcs["press_tank"] = {"P": Pcopv, "T": Tcopv}

        fluid_props = {
            "ox": {
                "rho": rho_ox_liq,
                "T": T_ox_liq,
                "h": h_ox_liq,
            },
            "fuel": {
                "rho": rho_fuel_liq,
                "T": T_fuel_liq,
                "h": h_fuel_liq,
            },
            "gas": {
                "rho": Pcopv,
                "T": Tcopv,
                "h": h_copv,
            },
        }

        net_out = self.network.resolve_network(
            bcs=bcs,
            fluid_props=fluid_props,
        )

        mdot_ox_out = net_out["mdot"].get("OX_INJ", 0.0)
        mdot_fuel_out = net_out["mdot"].get("FUEL_INJ", 0.0)

        mdot_ox_press = 0.0
        mdot_fuel_press = 0.0
 
        mdot_press_total = mdot_ox_press + mdot_fuel_press

        Q_copv = Qdot.get("press_tank", 0.0) if self._has_press_tank() else 0.0
        Q_ox_liq = Qdot.get("ox_liquid", 0.0)
        Q_ox_ull = Qdot.get("ox_ullage", 0.0)
        Q_fuel_liq = Qdot.get("fuel_liquid", 0.0)
        Q_fuel_ull = Qdot.get("fuel_ullage", 0.0)

        R = []

        if self._has_press_tank():
            R_copv_mass = (trial["copv_m"] - prev["copv_m"]) + dt * mdot_press_total
            R_copv_energy = (trial["copv_U"] - prev["copv_U"]) - dt * (
                Q_copv - mdot_press_total * h_copv
            )
            R.extend([R_copv_mass, R_copv_energy])

        R_ox_liq_mass = (trial["ox_liq_m"] - prev["ox_liq_m"]) + dt * mdot_ox_out
        R_ox_liq_energy = (trial["ox_liq_U"] - prev["ox_liq_U"]) - dt * (
            Q_ox_liq - mdot_ox_out * h_ox_liq
        )

        R_ox_ull_mass = (trial["ox_ull_m"] - prev["ox_ull_m"]) - dt * mdot_ox_press
        R_ox_ull_energy = (trial["ox_ull_U"] - prev["ox_ull_U"]) - dt * (
            Q_ox_ull + mdot_ox_press * h_copv
        )

        R_fuel_liq_mass = (trial["fuel_liq_m"] - prev["fuel_liq_m"]) + dt * mdot_fuel_out
        R_fuel_liq_energy = (trial["fuel_liq_U"] - prev["fuel_liq_U"]) - dt * (
            Q_fuel_liq - mdot_fuel_out * h_fuel_liq
        )

        R_fuel_ull_mass = (trial["fuel_ull_m"] - prev["fuel_ull_m"]) - dt * mdot_fuel_press
        R_fuel_ull_energy = (trial["fuel_ull_U"] - prev["fuel_ull_U"]) - dt * (
            Q_fuel_ull + mdot_fuel_press * h_copv
        )

        R.extend([
            R_ox_liq_mass,
            R_ox_liq_energy,
            R_ox_ull_mass,
            R_ox_ull_energy,
            R_fuel_liq_mass,
            R_fuel_liq_energy,
            R_fuel_ull_mass,
            R_fuel_ull_energy,
        ])

        R = np.array(R, dtype=float)

        result = {
            "ox_tank_state": {
                "liquid": {
                    "m": trial["ox_liq_m"],
                    "U": trial["ox_liq_U"],
                    "rho": rho_ox_liq,
                    "P": Pox,
                    "T": T_ox_liq,
                    "u": u_ox_liq_spec,
                    "h": h_ox_liq,
                    "V": V_ox_liq,
                    "Qdot": Q_ox_liq,
                },
                "ullage": {
                    "m": trial["ox_ull_m"],
                    "U": trial["ox_ull_U"],
                    "rho": rho_ox_ull,
                    "P": Pox,
                    "T": Tox,
                    "u": u_ox_gas_spec,
                    "h": h_ox_gas,
                    "V": V_ox_ull,
                    "Qdot": Q_ox_ull,
                },
            },

            "fuel_tank_state": {
                "liquid": {
                    "m": trial["fuel_liq_m"],
                    "U": trial["fuel_liq_U"],
                    "rho": rho_fuel_liq,
                    "P": Pfuel,
                    "T": T_fuel_liq,
                    "u": u_fuel_liq_spec,
                    "h": h_fuel_liq,
                    "V": V_fuel_liq,
                    "Qdot": Q_fuel_liq,
                },
                "ullage": {
                    "m": trial["fuel_ull_m"],
                    "U": trial["fuel_ull_U"],
                    "rho": rho_fuel_ull,
                    "P": Pfuel,
                    "T": Tfuel,
                    "u": u_fuel_gas_spec,
                    "h": h_fuel_gas,
                    "V": V_fuel_ull,
                    "Qdot": Q_fuel_ull,
                },
            }
        }

        return R, result
    

