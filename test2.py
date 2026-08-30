from Layout import Network, Component, State, Vehicle, Section, Sizer, KeroLOXEngineSizer, KeroLOXEngine
from CONFIGURATION import *
import fullplot as fplt

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
EngineSection = S



# ---- Vehicle Architecture ---- #
Elytra = Vehicle("Elytra")
PropSystem = Network("Propellant Feed System", Elytra)

chamber_pressure = 250 * PSIA_TO_PA

Engine = KeroLOXEngine(
    "Ares Ablative",
    PropSystem,
    chamber_pressure=chamber_pressure,
    chamber_volume=1,
    throat_area=1,
    expansion_ratio=1,
    ambient_pressure=14.67 * PSIA_TO_PA,
    nfz=2,
    fuel_mass_flow=2,
    oxidizer_mass_flow=4,
    engine_mass=50*LBM_TO_KG,
    sizer=engine_sizer
)



# ---- Size and Solve ----
Elytra.size()
print(Engine)
sol = Elytra.fly(dt=0.0005, t_final=0.1)
print(Engine)


'''
# ---- Plotting ---- #
pressure_trace = fplt.Trace(
    x=sol["time"],
    y=sol["Ares Ablative"]["chamber_pressure"],
    name="Volume pressure",
)

fplt.plot(
    [pressure_trace],
    xlabel="Time (s)",
    ylabel="Pressure (Pa)",
    title="Volume Pressure",
)

fplt.show()
'''