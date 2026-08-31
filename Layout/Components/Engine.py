from __future__ import annotations
from ..State import State
from ..Component import Component
from ..Sizer import Sizer
from typing import TYPE_CHECKING

from Physics import CEA

if TYPE_CHECKING:
    from ..Network import Network


class Engine(Component):

    def __init__(self, 
                 name: str, 
                 network: Network,
                 fuel:str, 
                 oxidizer:str,
                 chamber_pressure,
                 chamber_volume,
                 throat_area,
                 expansion_ratio,
                 ambient_pressure,
                 mass,
                 characteristic_velocity_efficiency = 1.0,
                 thrust_coefficient_efficiency = 1.0,
                 nfz = 0,
                 fuel_mass_flow = None,
                 oxidizer_mass_flow = None,
                 mixture_ratio = None,
                 thrust = None,
                 sizer: Sizer = None):
        self.setup()
        self.gas_mass = State()
        self.nozzle_mass_flow = State()

    def evaluate(self):
        Pc = self.chamber_pressure.value
        fuel_mdot = self.fuel_mass_flow.value
        ox_mdot = self.oxidizer_mass_flow.value
        V = self.chamber_volume.value
        At = self.throat_area.value
        eta_cstar = self.characteristic_velocity_efficiency.value
        eta_cf = self.thrust_coefficient_efficiency.value
        Pamb = self.ambient_pressure.value
        eps = self.expansion_ratio.value
        nfz = self.nfz.value
        fuel = self.fuel.value
        ox = self.oxidizer.value

        MR = ox_mdot / fuel_mdot
        self.mixture_ratio.value = MR
        cea = CEA(
            fuel=fuel,
            oxidizer=ox,
            chamber_pressure=Pc,
            mixture_ratio=MR,
            ambient_pressure=Pamb,
            expansion_ratio=eps,
            nfz=nfz,
        )
        self.gas_mass.value = cea.chamber_density * V

        self.nozzle_mass_flow.value = Pc * At / (
            cea.characteristic_velocity * eta_cstar
        )

        self.thrust.value = cea.thrust_coefficient * eta_cf * Pc * At


    @property
    def dynamics(self):
        return[(self.chamber_pressure, self.gas_mass, self.fuel_mass_flow.value + self.oxidizer_mass_flow.value - self.nozzle_mass_flow.value)]







class EngineSizer(Sizer):

    def __init__(self,
                 fuel:str, 
                 oxidizer:str,
                 chamber_pressure,
                 mixture_ratio,
                 thrust,
                 exit_pressure,
                 chamber_length,
                 contraction_ratio,
                 thrust_coefficient_efficiency = 1.0,
                 nfz = 0):
        self.setup()

    def size(self, engine: Engine):
        Pc = self.chamber_pressure
        MR = self.mixture_ratio
        F = self.thrust
        Pe = self.exit_pressure # assume perfectly expanded
        L = self.chamber_length
        eps_c = self.contraction_ratio
        eta_cf = self.thrust_coefficient_efficiency
        nfz = self.nfz
        fuel = self.fuel
        ox = self.oxidizer

        eps = CEA.calculate_expansion_ratio(fuel, ox, Pc, MR, Pe, nfz)
        cea = CEA(
            fuel=fuel,
            oxidizer=ox,
            chamber_pressure=Pc,
            mixture_ratio=MR,
            ambient_pressure=Pe,
            expansion_ratio=eps,
            nfz=nfz,
        )
        cf = cea.thrust_coefficient

    
        engine.expansion_ratio.value = eps
        engine.throat_area.value = F / (eta_cf * cf * Pc)
        engine.chamber_volume.value = eps_c * engine.throat_area.value * L
        engine.thrust_coefficient_efficiency.value = eta_cf
