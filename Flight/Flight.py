from __future__ import annotations
import math
from typing import Any, Dict, Optional

from types import KinematicsState, PlantOut
from types import AtmosState, AeroOut, ThermalOut, FluidOut, PlantOut
import PropSystem

class FlightSim:

    def __init__(self, cfg, env, aero, thermal, fluids, vehicle):
        self.cfg = cfg
        self.env = env
        self.aero = aero
        self.thermal = thermal
        self.fluids = fluids
        self.vehicle = vehicle

    def on_rail(self, h: float) -> bool:
        L = self.cfg.launch.rail_length
        h0 = self.cfg.launch.altitude
        return h < (h0 + L)

    def step_plant_coupled(self, kin: KinematicsState, atm: AtmosState, aero_out: AeroOut) -> PlantOut:
        dt = kin.dt
        couple = self.cfg.sim.coupling

        Tguess = self.thermal.get_wall_temp()
        Pguess = self.fluids.get_pressures()

        thermal_out: Optional[ThermalOut] = None
        fluid_out: Optional[FluidOut] = None

        for _it in range(couple.max_iters):
            fluids_bc = self.fluids.get_internal_thermal_bc()

            thermal_out = PropSystem.step(
                dt=dt,
                heat_bc=aero_out.heat_bc,
                fluids_bc=fluids_bc,
                wall_T0=Tguess,
            )
            Tnew = thermal_out.wall_T

            fluid_out = self.fluids.step(
                dt=dt,
                kin=kin,
                atm=atm,
                thermal_out=thermal_out,
                aero_out=aero_out,
            )
            Pnew = fluid_out.Pvec

            dP = max(abs(Pnew[0] - Pguess[0]), abs(Pnew[1] - Pguess[1]), abs(Pnew[2] - Pguess[2]))
            dT = abs(float(Tnew) - float(Tguess))

            Pguess = Pnew
            Tguess = Tnew

            if dP < couple.tol_P and dT < couple.tol_T:
                break

        assert thermal_out is not None and fluid_out is not None
        return PlantOut(aero=aero_out, thermal=thermal_out, fluids=fluid_out)

    def step_kinematics_RK4(self, kin: KinematicsState, plant: PlantOut) -> KinematicsState:
        dt = 
        g = 

        W = 
        F = 
        D = 

        a = 
        v1 = 
        h1 = 

        return KinematicsState(
            t=kin.t + dt,
            dt=dt,
            h=h1,
            v=v1,
            w=kin.w,         
            alpha=kin.alpha,
            m=kin.m,
            Ixx=kin.Ixx,
        )

    def run(self, h0: float = 0.0, v0: float = 0.01) -> None:
        
        dt = self.cfg.sim.dt

        self.vehicle.mass()

        kin = KinematicsState(
            t=0.0, dt=dt,
            h=h0, v=v0, w=0.0, alpha=0.0,
            m=self.vehicle.mass, Ixx=self.vehicle.Ixx
        )

        while kin.t < self.cfg.sim.t_end and kin.v >= 0.0:

            self.vehicle.update_mass_vector() #mass vec
            self.vehicle.update_mass_properties() #total mass, COG, MOI, etc.

            atm = self.env.atmosphere(kin.h, kin.v)

            kin.alpha = self.aero.aoa(kin, atm, self.on_rail(kin.h))

            aero_out = self.aero.evaluate(kin, atm)

            plant = self.step_plant_coupled(kin, atm, aero_out)

            dm_by_tank = {
                "fuel_tank": plant.fluids.dm_fuel,
                "ox_tank": plant.fluids.dm_ox,
            }
            self.vehicle.tankdrain(dm_by_tank)

            kin.m = self.vehicle.total_mass
            kin.Ixx = self.vehicle.Ixx

            kin = self.step_kinematics_RK4(kin, plant)

            # TODO: log