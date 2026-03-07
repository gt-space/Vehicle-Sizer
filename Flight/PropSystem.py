import numpy as np
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable
from .types import KinematicsState, AtmosState, ThermalOut, AeroOut, FluidOut
import FluidNetwork

class PropSystem:

    def __init__(self, cfg: Dict[str, any]):

        self.cfg = cfg
        self.model = cfg["press_model"]
        self._init_model(cfg) 
        self.target_ladder = self._build_pressure_ladder(cfg)


        nodes, branches, params = self._wire_network(cfg)

        self._size_network(nodes, branches, cfg)

        self.network = FluidNetwork(
            nodes=nodes,
            branches=branches,
            params=params,
        )

    def _wire_network(self, cfg) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:

        if self.model == "pump_fed":
            nodes, branches = self._template_pump_fed(cfg)
            return nodes, branches

        elif self.model == "pressure_fed":
            nodes, branches = self._template_pressure_fed(cfg)
            return nodes, branches

        elif self.model == "blowdown":
            nodes, branches = self._template_blowdown(cfg)
            return nodes, branches

        else: raise ValueError(f"Unknown press_model={self.model}")
    
    def _template_pump_fed(self, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:

        nodes: List[Dict[str, Any]] = [
            {"id": "press_tank", "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "ox_ullage",   "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "fuel_ullage", "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "ox_pump_in", "type": "liquid_junction"},
            {"id": "ox_pump_out", "type": "liquid_junction"},
            {"id": "ox_inj_in", "type": "liquid_junction"},
            {"id": "fuel_pump_in", "type": "liquid_junction"},
            {"id": "fuel_pump_out", "type": "liquid_junction"},
            {"id": "fuel_inj_in", "type": "liquid_junction"},
            {"id": "thrust_chamber", "type": "chamber"},
            {"id": "ambient", "type": "boundary_pressure"},
        ]

        branches: List[Dict[str, Any]] = [
            {"id": "OX_BANGBANG", "type": "gas_orifice", "from": "press_tank", "to": "ox_ullage", "CdA": None},
            {"id": "FUEL_BANGBANG", "type": "gas_orifice", "from": "press_tank", "to": "fuel_ullage", "CdA": None},
            {"id": "OX_TANK_PUMP", "type": "liquid_loss", "from": "ox_ullage", "to": "ox_pump_in", "CdA": None},
            {"id": "OX_PUMP", "type": "pump", "from": "ox_pump_in", "to": "ox_pump_out", "dP": None},
            {"id": "OX_PUMP_INJ", "type": "liquid_loss", "from": "ox_pump_out", "to": "ox_inj_in", "CdA": None},
            {"id": "OX_INJ", "type": "injector", "from": "ox_inj_in", "to": "thrust_chamber", "CdA": None},
            {"id": "FUEL_TANK_PUMP", "type": "liquid_loss", "from": "fuel_ullage", "to": "fuel_pump_in", "CdA": None},
            {"id": "FUEL_PUMP", "type": "pump", "from": "fuel_pump_in", "to": "fuel_pump_out", "dP": None},
            {"id": "FUEL_PUMP_INJ", "type": "liquid_loss", "from": "fuel_pump_out", "to": "fuel_inj_in", "CdA": None},
            {"id": "FUEL_INJ", "type": "injector", "from": "fuel_inj_in", "to": "thrust_chamber", "CdA": None},
            {"id": "Nozzle", "type": "gas_orifice", "from": "thrust_chamber", "to": "ambient", "CdA": None}
        ]

        return nodes, branches
    
    def _template_pressure_fed(self, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:

        nodes: List[Dict[str, Any]] = [
            {"id": "press_tank", "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "ox_ullage",   "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "fuel_ullage", "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "ox_inj_in", "type": "liquid_junction"},
            {"id": "fuel_inj_in", "type": "liquid_junction"},
            {"id": "thrust_chamber", "type": "chamber"},
            {"id": "ambient", "type": "boundary_pressure"},
        ]

        branches: List[Dict[str, Any]] = [

            {"id": "OX_BANGBANG", "type": "gas_orifice", "from": "press_tank", "to": "ox_ullage", "CdA": None},
            {"id": "FUEL_BANGBANG", "type": "gas_orifice", "from": "press_tank", "to": "fuel_ullage", "CdA": None},
            {"id": "OX_TANK_INJ", "type": "liquid_loss", "from": "ox_ullage", "to": "ox_inj_in", "CdA": None},
            {"id": "OX_INJ", "type": "injector", "from": "ox_inj_in", "to": "thrust_chamber", "CdA": None},
            {"id": "FUEL_TANK_INJ", "type": "liquid_loss", "from": "fuel_ullage", "to": "fuel_inj_in", "CdA": None},
            {"id": "FUEL_INJ", "type": "injector", "from": "fuel_inj_in", "to": "thrust_chamber", "CdA": None},
            {"id": "Nozzle", "type": "gas_orifice", "from": "thrust_chamber", "to": "ambient", "CdA": None}
        ]

        return nodes, branches
    
        # each branch/node dict should call size function? Sizing branches and nodes? For nodes might need to call on geometry classes
        # sections for prop tanks and press tanks should get passed into here to assign for node
        # each type should have a call 
        # 
    
    def _template_blowdown(self, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:

        nodes: List[Dict[str, Any]] = [
            {"id": "ox_ullage",   "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "fuel_ullage", "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "ox_inj_in", "type": "liquid_junction"},
            {"id": "fuel_inj_in", "type": "liquid_junction"},
            {"id": "thrust_chamber", "type": "chamber"},
            {"id": "ambient", "type": "boundary_pressure"},
        ]

        branches: List[Dict[str, Any]] = [

            {"id": "OX_TANK_INJ", "type": "liquid_loss", "from": "ox_ullage", "to": "ox_inj_in", "CdA": None},
            {"id": "OX_INJ", "type": "injector", "from": "ox_inj_in", "to": "thrust_chamber", "CdA": None},
            {"id": "FUEL_TANK_INJ", "type": "liquid_loss", "from": "fuel_ullage", "to": "fuel_inj_in", "CdA": None},
            {"id": "FUEL_INJ", "type": "injector", "from": "fuel_inj_in", "to": "thrust_chamber", "CdA": None},
            {"id": "Nozzle", "type": "gas_orifice", "from": "thrust_chamber", "to": "ambient", "CdA": None}
        ]

        return nodes, branches

    def _init_model(self, cfg: Dict[str, Any]) -> None:
        # initialize from config based on model
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
        if self.model == "pressure_fed":
            self.Pc_target = cfg['Pc_target']
            self.fuel_inj_stiffness = cfg["fuel_inj_stiffness"]
            self.ox_inj_stiffness = cfg["ox_inj_stiffness"]
            self.fuel_tank_inj_dp = cfg["fuel_tank_inj_dp"]
            self.ox_tank_inj_dp = cfg["ox_tank_inj_dp"]
            self.ox_pressurant_dp = cfg["ox_pressurant_dp"]
            self.fuel_pressurant_dp = cfg["fuel_pressurant_dp"]
        if self.model == "blowdown":
            self.Pc_target = cfg['Pc_target']
            self.fuel_inj_stiffness = cfg["fuel_inj_stiffness"]
            self.ox_inj_stiffness = cfg["ox_inj_stiffness"]
            self.fuel_tank_inj_dp = cfg["fuel_tank_inj_dp"]
            self.ox_tank_inj_dp = cfg["ox_tank_inj_dp"]
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

    def _wire_network(self, cfg) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:

        if self.model == "pump_fed":
            nodes, branches = self._template_pump_fed(cfg)
            return nodes, branches

        if self.model == "pressure_fed":
            nodes, branches = self._template_pressure_fed(cfg)
            return nodes, branches

        if self.model == "blowdown":
            nodes, branches = self._template_blowdown(cfg)
            return nodes, branches

        raise ValueError(f"Unknown press_model={self.model}")
    
    def _template_pump_fed(self, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:

        nodes: List[Dict[str, Any]] = [
            {"id": "press_tank", "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "ox_ullage",   "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "fuel_ullage", "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "ox_pump_in", "type": "liquid_junction"},
            {"id": "ox_pump_out", "type": "liquid_junction"},
            {"id": "ox_inj_in", "type": "liquid_junction"},
            {"id": "fuel_pump_in", "type": "liquid_junction"},
            {"id": "fuel_pump_out", "type": "liquid_junction"},
            {"id": "fuel_inj_in", "type": "liquid_junction"},
            {"id": "thrust_chamber", "type": "chamber"},
            {"id": "ambient", "type": "boundary_pressure"},
        ]

        branches: List[Dict[str, Any]] = [
            {"id": "OX_BANGBANG", "type": "gas_orifice", "from": "press_tank", "to": "ox_ullage", "CdA": None},
            {"id": "FUEL_BANGBANG", "type": "gas_orifice", "from": "press_tank", "to": "fuel_ullage", "CdA": None},
            {"id": "OX_TANK_PUMP", "type": "liquid_loss", "from": "ox_ullage", "to": "ox_pump_in", "CdA": None},
            {"id": "OX_PUMP", "type": "pump", "from": "ox_pump_in", "to": "ox_pump_out", "dP": None},
            {"id": "OX_PUMP_INJ", "type": "liquid_loss", "from": "ox_pump_out", "to": "ox_inj_in", "CdA": None},
            {"id": "OX_INJ", "type": "injector", "from": "ox_inj_in", "to": "thrust_chamber", "CdA": None},
            {"id": "FUEL_TANK_PUMP", "type": "liquid_loss", "from": "fuel_ullage", "to": "fuel_pump_in", "CdA": None},
            {"id": "FUEL_PUMP", "type": "pump", "from": "fuel_pump_in", "to": "fuel_pump_out", "dP": None},
            {"id": "FUEL_PUMP_INJ", "type": "liquid_loss", "from": "fuel_pump_out", "to": "fuel_inj_in", "CdA": None},
            {"id": "FUEL_INJ", "type": "injector", "from": "fuel_inj_in", "to": "thrust_chamber", "CdA": None},
            {"id": "Nozzle", "type": "gas_orifice", "from": "thrust_chamber", "to": "ambient", "CdA": None}
        ]

        return nodes, branches
    
    def _template_pressure_fed(self, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:

        nodes: List[Dict[str, Any]] = [
            {"id": "press_tank", "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "ox_ullage",   "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "fuel_ullage", "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "ox_inj_in", "type": "liquid_junction"},
            {"id": "fuel_inj_in", "type": "liquid_junction"},
            {"id": "thrust_chamber", "type": "chamber"},
            {"id": "ambient", "type": "boundary_pressure"},
        ]

        branches: List[Dict[str, Any]] = [

            {"id": "OX_BANGBANG", "type": "gas_orifice", "from": "press_tank", "to": "ox_ullage", "CdA": None},
            {"id": "FUEL_BANGBANG", "type": "gas_orifice", "from": "press_tank", "to": "fuel_ullage", "CdA": None},
            {"id": "OX_TANK_INJ", "type": "liquid_loss", "from": "ox_ullage", "to": "ox_inj_in", "CdA": None},
            {"id": "OX_INJ", "type": "injector", "from": "ox_inj_in", "to": "thrust_chamber", "CdA": None},
            {"id": "FUEL_TANK_INJ", "type": "liquid_loss", "from": "fuel_ullage", "to": "fuel_inj_in", "CdA": None},
            {"id": "FUEL_INJ", "type": "injector", "from": "fuel_inj_in", "to": "thrust_chamber", "CdA": None},
            {"id": "Nozzle", "type": "gas_orifice", "from": "thrust_chamber", "to": "ambient", "CdA": None}
        ]

        return nodes, branches
    
        # each branch/node dict should call size function? Sizing branches and nodes? For nodes might need to call on geometry classes
        # sections for prop tanks and press tanks should get passed into here to assign for node
        # each type should have a call 
        # 
    
    def _template_blowdown(self, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:

        nodes: List[Dict[str, Any]] = [
            {"id": "ox_ullage",   "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "fuel_ullage", "type": "gas_volume", "V": None, "state0": {"P": None, "T": None}},
            {"id": "ox_inj_in", "type": "liquid_junction"},
            {"id": "fuel_inj_in", "type": "liquid_junction"},
            {"id": "thrust_chamber", "type": "chamber"},
            {"id": "ambient", "type": "boundary_pressure"},
        ]

        branches: List[Dict[str, Any]] = [

            {"id": "OX_TANK_INJ", "type": "liquid_loss", "from": "ox_ullage", "to": "ox_inj_in", "CdA": None},
            {"id": "OX_INJ", "type": "injector", "from": "ox_inj_in", "to": "thrust_chamber", "CdA": None},
            {"id": "FUEL_TANK_INJ", "type": "liquid_loss", "from": "fuel_ullage", "to": "fuel_inj_in", "CdA": None},
            {"id": "FUEL_INJ", "type": "injector", "from": "fuel_inj_in", "to": "thrust_chamber", "CdA": None},
            {"id": "Nozzle", "type": "gas_orifice", "from": "thrust_chamber", "to": "ambient", "CdA": None}
        ]

        return nodes, branches

    def _init_model(self, cfg: Dict[str, Any]) -> None:
        # initialize from config based on model
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
        if self.model == "pressure_fed":
            self.Pc_target = cfg['Pc_target']
            self.fuel_inj_stiffness = cfg["fuel_inj_stiffness"]
            self.ox_inj_stiffness = cfg["ox_inj_stiffness"]
            self.fuel_tank_inj_dp = cfg["fuel_tank_inj_dp"]
            self.ox_tank_inj_dp = cfg["ox_tank_inj_dp"]
            self.ox_pressurant_dp = cfg["ox_pressurant_dp"]
            self.fuel_pressurant_dp = cfg["fuel_pressurant_dp"]
        if self.model == "blowdown":
            self.Pc_target = cfg['Pc_target']
            self.fuel_inj_stiffness = cfg["fuel_inj_stiffness"]
            self.ox_inj_stiffness = cfg["ox_inj_stiffness"]
            self.fuel_tank_inj_dp = cfg["fuel_tank_inj_dp"]
            self.ox_tank_inj_dp = cfg["ox_tank_inj_dp"]
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
        if self.model == "pressure_fed" or "blowdown":
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
        
    def _size_branches(self, branches: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
        pass

    def _size_nodes(self, nodes: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
        pass
    
    ## make ts better
    def get_pressures(self) -> Tuple[float, float, float]:
        return (self._Pox, self._Pfuel, self._Pcopv)

    def get_internal_thermal_bc(self) -> Dict[str, Any]:
        return {}

    def step(self, dt, kin, atm, thermal_out, aero_out) -> FluidOut:
        
        Qdot = thermal_out.Qdot  

        self.get_valve_state()  

        # This is a DAE system where the differential transient phenomena are the COPV and tank ullage states 
        # and the algebraic terms are the pressures and mass flows at each timestep. Implicit newton is used for
        # solving the states while simeltaneously solving the algebraic root solve. State is therefore rho/U for COPV,
        # P,T for ullage, and masses in the tanks 

        x_n = np.array([

            self.copv.rho,       
            self.copv.U,           

            self.ox_tank.prop_mass,
            self.ox_tank.rho,  
            self.ox_tank.U,   

            self.fuel_tank.prop_mass,
            self.fuel_tank.rho,
            self.fuel_tank.U,  

        ], dtype=float)

        cache = {"result": None}

        def fun(x_guess: np.ndarray) -> np.ndarray:
            R, result = self.residual(
                x_guess=x_guess,
                x_prev=x_n,
                dt=dt,
                Qdot=Qdot, # placeholder 
            )
            cache["result"] = result
            return R

        sol = root(fun, x_n, method="hybr") 
        if not sol.success:
            raise RuntimeError(f"Implicit solve failed: {sol.message}")

        x_np1 = sol.x
        result = cache["result"]

        self.copv.rho = x_np1[0]
        self.copv.U = x_np1[1]
        self.ox_tank.prop_mass = x_np1[2]
        self.ox_tank.rho = x_np1[3]
        self.ox_tank.U = x_np1[4]
        self.fuel_tank.prop_mass = x_np1[5]
        self.fuel_tank.rho = x_np1[6]
        self.fuel_tank.U = x_np1[7]

        return x_np1, result
    
    def residual(self, x_guess: np.ndarray, x_prev: np.ndarray, dt: float, Qdot: Dict[str, float]):

        rho_c_n = float(x_guess[0])
        U_c_n   = float(x_guess[1])

        mox_n   = float(x_guess[2])
        Pox_n   = float(x_guess[3])
        Tox_n   = float(x_guess[4])

        mfuel_n = float(x_guess[5])
        Pfuel_n = float(x_guess[6])
        Tfuel_n = float(x_guess[7])

        rho_c = float(x_prev[0])
        U_c   = float(x_prev[1])

        mox   = float(x_prev[2])
        Pox   = float(x_prev[3])
        Tox   = float(x_prev[4])

        mfuel = float(x_prev[5])
        Pfuel = float(x_prev[6])
        Tfuel = float(x_prev[7])

        
