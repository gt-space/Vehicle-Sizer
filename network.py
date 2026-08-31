from Layout import *
from CONFIGURATION import *

ElytraLaunch = LaunchInputs(initial_altitude=INITIAL_ALTITUDE)

Elytra = Vehicle("Elytra", ElytraLaunch)
ElytraFeedSystem = Network("Elytra Feed System", Elytra)
'''
ElytraCoax = Injector(
    "Elytra Coax",
    ElytraFeedSystem,
    #fuel_propellant=

)'''

from Physics import Fluid
from thermoprop import Propellant

print(Fluid("nDodecane", pressure=101325, temperature=300).viscosity)
print(Propellant("RP-1", pressure=101325, temperature=300).dynamic_viscosity)
