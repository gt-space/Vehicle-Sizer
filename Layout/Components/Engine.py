from __future__ import annotations
from ..State import State
from ..Component import Component
from ..Sizer import Sizer
from typing import TYPE_CHECKING

from Physics import get_cf_ideal, get_eps, get_chamber_density, get_cstar_ideal

if TYPE_CHECKING:
    from ..Network import Network
    from ..Sizer import Sizer


class KeroLOXEngine(Component):

    def __init__(self, 
                 name: str, 
                 network: Network,
                 chamber_pressure,
                 chamber_volume,
                 throat_area,
                 expansion_ratio,
                 ambient_pressure,
                 engine_mass,
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

        MR = ox_mdot / fuel_mdot
        self.mixture_ratio.value = MR
        rho = get_chamber_density(Pc=Pc, MR=MR)
        self.gas_mass.value = rho*V

        cstar_ideal = get_cstar_ideal(Pc=Pc, MR=MR)
        self.nozzle_mass_flow.value = Pc * At / (cstar_ideal * eta_cstar)

        cf_ideal = get_cf_ideal(Pc=Pc, MR=MR, Pamb=Pamb, eps=eps, nfz=nfz)
        self.thrust.value = cf_ideal * eta_cf * Pc * At


    @property
    def dynamics(self):
        return[(self.chamber_pressure, self.gas_mass, self.fuel_mass_flow.value + self.oxidizer_mass_flow.value - self.nozzle_mass_flow.value)]







class KeroLOXEngineSizer(Sizer):

    def __init__(self,
                 chamber_pressure,
                 mixture_ratio,
                 thrust,
                 exit_pressure,
                 chamber_length,
                 contraction_ratio,
                 thrust_coefficient_efficiency = 1.0,
                 nfz = 0):
        self.setup()

    def size(self, engine: KeroLOXEngine):
        Pc = self.chamber_pressure
        MR = self.mixture_ratio
        F = self.thrust
        Pe = self.exit_pressure # assume perfectly expanded
        L = self.chamber_length
        eps_c = self.contraction_ratio
        eta_cf = self.thrust_coefficient_efficiency
        nfz = self.nfz

        eps = get_eps(Pc=Pc, MR=MR, Pe=Pe, nfz=nfz)
        cf = get_cf_ideal(Pc=Pc, MR=MR, Pamb=Pe, eps=eps, nfz=nfz)

    
        engine.expansion_ratio.value = eps
        engine.throat_area.value = F / (eta_cf * cf * Pc)
        engine.chamber_volume.value = eps_c * engine.throat_area.value * L
        engine.thrust_coefficient_efficiency.value = eta_cf