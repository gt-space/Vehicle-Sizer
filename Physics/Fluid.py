from CoolProp.CoolProp import PropsSI


class Fluid:

    def __init__(self, 
                 name: str, 
                 pressure: float, 
                 temperature: float = None, 
                 enthalpy: float = None):
        
        if (temperature is None) == (enthalpy is None):
            raise ValueError("Provide either temperature or enthalpy, but not both.")

        self.name = name
        self._pressure = float(pressure)
        if temperature is not None:
            self._input = "T"
            self._value = float(temperature)
        else:
            self._input = "Hmass"
            self._value = float(enthalpy)

    def _get(self, property_name: str) -> float:
        return PropsSI(property_name, "P", self._pressure, self._input, self._value, self.name)

    @property
    def pressure(self) -> float:
        return self._pressure

    @property
    def temperature(self) -> float:
        return self._get("T")

    @property
    def enthalpy(self) -> float:
        return self._get("Hmass")

    @property
    def density(self) -> float:
        return self._get("Dmass")

    @property
    def entropy(self) -> float:
        return self._get("Smass")

    @property
    def internal_energy(self) -> float:
        return self._get("Umass")

    @property
    def specific_heat_cp(self) -> float:
        return self._get("Cpmass")

    @property
    def specific_heat_cv(self) -> float:
        return self._get("Cvmass")

    @property
    def viscosity(self) -> float:
        return self._get("V")

    @property
    def thermal_conductivity(self) -> float:
        return self._get("L")

    @property
    def speed_of_sound(self) -> float:
        return self._get("A")

    @property
    def quality(self) -> float:
        return self._get("Q")
