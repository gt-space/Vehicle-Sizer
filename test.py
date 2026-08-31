from Layout import Network, Component, State, Vehicle, Section, Sizer, KeroLOXEngineSizer, KeroLOXEngine, LaunchInputs
from CONFIGURATION import *
import matplotlib.pyplot as plt

# ---- Sizers ---- #
engine_sizer = KeroLOXEngineSizer(
    chamber_pressure=CHAMBER_PRESSURE,
    mixture_ratio=MIXTURE_RATIO,
    thrust=THRUST,
    exit_pressure=EXIT_PRESSURE,
    chamber_length=CHAMBER_LENGTH,
    contraction_ratio=CONTRACTION_RATIO,
    thrust_coefficient_efficiency=THRUST_COEFFICIENT_EFFICIENCY,
    nfz=NFZ
)



# ---- New Section (if needed) ---- #
class Boattail(Component): 
    def __init__(self, name, network, mass):
        self.setup()


class EngineSection(Section):

    def __init__(self, 
                 name: str, 
                 vehicle: Vehicle,
                 engine: KeroLOXEngine,
                 boattail: Boattail):
        self.setup()

    def evaluate(self):
        self.mass.value = self.engine.mass.value + self.boattail.mass.value
        self.length.value = 20 * IN_TO_M
        self.EI.value = 1.0


# ---- Laucnh Inputs ---- #
ElytraLaunch = LaunchInputs(initial_altitude=INITIAL_ALTITUDE)


# ---- Vehicle Architecture ---- #
Elytra = Vehicle("Elytra", ElytraLaunch)
PropSystem = Network("Propellant Feed System", Elytra)

chamber_pressure = 250 * PSIA_TO_PA

Engine = KeroLOXEngine(
    "Ares Ablative",
    PropSystem,
    chamber_pressure=chamber_pressure,
    chamber_volume=1,
    throat_area=1,
    expansion_ratio=1,
    ambient_pressure=Elytra.atmospheric_pressure,
    nfz=2,
    fuel_mass_flow=2,
    oxidizer_mass_flow=4,
    mass=50*LBM_TO_KG,
    thrust=Elytra.thrust,
    sizer=engine_sizer
)

FinCan = Boattail("Fin Can", PropSystem, mass=BOATTAIL_MASS)


ElyEngineSection = EngineSection(
    "Elytra Engine Section",
    Elytra,
    engine=Engine,
    boattail=FinCan
)

# ---- Size and Solve ----
Elytra.size()
print(Engine)
print(Elytra)
sol = Elytra.fly(dt=DT, t_final=T_FINAL)
print(Engine)
print(Elytra.thrust.value / LBF_TO_N)


'''
# ---- Plotting ---- #
time = sol['time']
Pc = [P / PSIA_TO_PA for P in sol["Ares Ablative"]["chamber_pressure"]]

plt.plot(time, Pc)
plt.show()
'''