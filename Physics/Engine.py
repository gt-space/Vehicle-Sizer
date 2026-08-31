from rocketcea.cea_obj_w_units import CEA_Obj


class CEA:


    def __init__(self, fuel:str, oxidizer: str, chamber_pressure, 
                 mixture_ratio, ambient_pressure=None, expansion_ratio=None, nfz=0):
        self.chamber_pressure = chamber_pressure
        self.mixture_ratio = mixture_ratio
        self.ambient_pressure = ambient_pressure
        self.expansion_ratio = expansion_ratio
        self.nfz = nfz

        self._cea = CEA_Obj(
            oxName=oxidizer,
            fuelName=fuel,
            temperature_units="degK",
            cstar_units="m/sec",
            specific_heat_units="kJ/kg degK",
            sonic_velocity_units="m/s",
            enthalpy_units="J/kg",
            density_units="kg/m^3",
            pressure_units="Pa",
        )

    @property
    def characteristic_velocity(self):
        return self._cea.get_Cstar(self.chamber_pressure, self.mixture_ratio)


    @property
    def chamber_density(self):
        return self._cea.get_Chamber_Density(self.chamber_pressure, self.mixture_ratio)


    @property
    def thrust_coefficient(self):
        if self.ambient_pressure is None or self.expansion_ratio is None:
            raise ValueError("ambient_pressure and expansion_ratio are required")

        arguments = (self.ambient_pressure, self.chamber_pressure,
                     self.mixture_ratio, self.expansion_ratio)
        if self.nfz == 2:
            _, coefficient, _ = self._cea.getFrozen_PambCf(*arguments, 1)
        elif self.nfz == 1:
            _, coefficient, _ = self._cea.getFrozen_PambCf(*arguments, 0)
        else:
            _, coefficient, _ = self._cea.get_PambCf(*arguments)
        return coefficient


    @staticmethod
    def calculate_expansion_ratio(fuel, oxidizer, chamber_pressure, mixture_ratio, exit_pressure, nfz=0):
        pressure_ratio = chamber_pressure / exit_pressure
        if nfz == 2:
            frozen, frozen_at_throat = 1, 1
        elif nfz == 1:
            frozen, frozen_at_throat = 1, 0
        else:
            frozen, frozen_at_throat = 0, 0

        return CEA._cea.get_eps_at_PcOvPe(
            chamber_pressure, mixture_ratio, pressure_ratio,
            frozen, frozen_at_throat)
