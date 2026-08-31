from Layout import *
from CONFIGURATION import *
from sizers import *
import matplotlib.pyplot as plt

ElytraLaunch = LaunchInputs(initial_altitude=INITIAL_ALTITUDE)

Elytra = Vehicle("Elytra", ElytraLaunch)
ElytraFeedSystem = Network("Elytra Feed System", Elytra)

chamber_pressure = State(CHAMBER_PRESSURE)

Coax = Injector(
    "Ares Coax",
    ElytraFeedSystem,
    fuel='RP-1',
    oxidizer='LOX',
    fuel_pressure=FUEL_MANIFOLD_PRESSURE,
    oxidizer_pressure=OXIDIZER_MANIFOLD_PRESSURE,
    fuel_temperature=FUEL_TEMPERATURE,
    oxidizer_temperature=OXIDIZER_TEMPERATURE,
    fuel_manifold_volume=FUEL_MANIFOLD_VOLUME,
    oxidizer_manifold_volume=OXIDIZER_MANIFOLD_VOLUME,
    fuel_flow_area=UNASSIGNED,
    oxidizer_flow_area=UNASSIGNED,
    chamber_pressure=chamber_pressure,
    mass=INJECTOR_MASS,
    fuel_mass_flow_in=FUEL_MASS_FLOW,
    oxidizer_mass_flow_in=OXIDIZER_MASS_FLOW,
    sizer=injector_sizer)


Ares = Engine(
    "Ares TCA",
    ElytraFeedSystem,
    fuel=FUEL,
    oxidizer=OXIDIZER,
    chamber_pressure=chamber_pressure,
    chamber_volume=UNASSIGNED,
    throat_area=UNASSIGNED,
    expansion_ratio=UNASSIGNED,
    ambient_pressure=Elytra.atmospheric_pressure,
    characteristic_velocity_efficiency=CHARACTERSTIC_VELOCITY_EFFICIENCY,
    thrust_coefficient_efficiency=THRUST_COEFFICIENT_EFFICIENCY,
    nfz=NFZ,
    mass=ENGINE_MASS,
    fuel_mass_flow=Coax.fuel_mass_flow_out,
    oxidizer_mass_flow=Coax.oxidizer_mass_flow_out,
    thrust=Elytra.thrust,
    sizer=engine_sizer
)



Elytra.size()
flight = Elytra.fly(dt=DT, t_final=T_FINAL)


# ---- Plotting ---- #
time = flight['time']
fuel_stiffness = flight["Ares Coax"]["fuel_stiffness"]

plt.plot(time, fuel_stiffness)
plt.show()