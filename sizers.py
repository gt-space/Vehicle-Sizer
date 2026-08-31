from Layout import EngineSizer, InjectorSizer
from CONFIGURATION import *

engine_sizer = EngineSizer(
    fuel=FUEL,
    oxidizer=OXIDIZER,
    chamber_pressure=CHAMBER_PRESSURE,
    mixture_ratio=MIXTURE_RATIO,
    thrust=THRUST,
    exit_pressure=EXIT_PRESSURE,
    chamber_length=CHAMBER_LENGTH,
    contraction_ratio=CONTRACTION_RATIO,
    thrust_coefficient_efficiency=THRUST_COEFFICIENT_EFFICIENCY,
    nfz=NFZ
)

injector_sizer = InjectorSizer(
    chamber_pressure=CHAMBER_PRESSURE,
    fuel_stiffness=FUEL_STIFFNESS,
    fuel_density=FUEL_DENSITY,
    fuel_mass_flow=FUEL_MASS_FLOW,
    oxidizer_stiffness=OXIDIZER_STIFFNESS,
    oxidizer_density=OXIDIZER_DENSITY,
    oxidizer_mass_flow=OXIDIZER_MASS_FLOW
)
