from __future__ import annotations
from ..State import State
from ..Component import Component
from ..Sizer import Sizer
from typing import TYPE_CHECKING

from Physics import Fluid, IncompressibleFlow

if TYPE_CHECKING:
    from ..Network import Network


class Injector(Component):

    def __init__(self, 
                 name: str, 
                 network: Network,
                 fuel: str,
                 oxidizer: str,
                 fuel_pressure,
                 oxidizer_pressure,
                 fuel_temperature,
                 oxidizer_temperature,
                 fuel_manifold_volume,
                 oxidizer_manifold_volume,
                 fuel_flow_area,
                 oxidizer_flow_area,
                 chamber_pressure,
                 mass,
                 fuel_mass_flow_in=None,
                 oxidizer_mass_flow_in=None,
                 fuel_stiffness=None,
                 oxidizer_stiffness=None,
                 sizer: Sizer = None):
        self.setup()
        self.fuel_fluid_mass = State()
        self.oxidizer_fluid_mass = State()
        self.fuel_mass_flow_out = State()
        self.oxidizer_mass_flow_out = State()

    def evaluate(self):
        Pc = self.chamber_pressure.value

        fuel_density = Fluid(name=self.fuel.value,
            pressure=self.fuel_pressure.value,
            temperature=self.fuel_temperature.value).density
        
        self.fuel_fluid_mass.value = fuel_density * self.fuel_manifold_volume.value
        self.fuel_mass_flow_out.value = IncompressibleFlow.mass_flow_from_cda(self.fuel_pressure.value, Pc, fuel_density, self.fuel_flow_area.value)

        oxidizer_density = Fluid(
            name=self.oxidizer.value,
            pressure=self.oxidizer_pressure.value,
            temperature=self.oxidizer_temperature.value).density
        self.oxidizer_fluid_mass.value = oxidizer_density * self.oxidizer_manifold_volume.value

        self.oxidizer_mass_flow_out.value = IncompressibleFlow.mass_flow_from_cda(self.oxidizer_pressure.value,
            Pc, oxidizer_density, self.oxidizer_flow_area.value)

        self.fuel_stiffness.value = (self.fuel_pressure.value - Pc) / Pc
        self.oxidizer_stiffness.value = (self.oxidizer_pressure.value - Pc) / Pc
        

    @property
    def dynamics(self):
        return [
            (self.fuel_pressure, self.fuel_fluid_mass, self.fuel_mass_flow_in.value - self.fuel_mass_flow_out.value),
            (self.oxidizer_pressure, self.oxidizer_fluid_mass, self.oxidizer_mass_flow_in.value - self.oxidizer_mass_flow_out.value),
        ]




class InjectorSizer(Sizer):

    def __init__(self,
                 chamber_pressure,
                 fuel_stiffness,
                 fuel_density,
                 fuel_mass_flow,
                 oxidizer_stiffness,
                 oxidizer_density,
                 oxidizer_mass_flow):
        self.setup()

    def size(self, injector: Injector):
        Pc = self.chamber_pressure
        stiff_fuel = self.fuel_stiffness
        rho_f = self.fuel_density
        mdot_f = self.fuel_mass_flow
        stiff_ox = self.oxidizer_stiffness
        rho_ox = self.oxidizer_density
        mdot_ox = self.oxidizer_mass_flow

        Pf = (1 + stiff_fuel) * Pc
        Pox = (1 + stiff_ox) * Pc
        injector.fuel_flow_area.value = IncompressibleFlow.cda_from_mass_flow(Pf, Pc, rho_f, mdot_f)
        injector.oxidizer_flow_area.value = IncompressibleFlow.cda_from_mass_flow(Pox, Pc, rho_ox, mdot_ox)
