from CoolProp.CoolProp import PropsSI
from rocketprops.rocket_prop import get_prop


ROCKETPROPS_FLUIDS = ["RP1"]
FLUID_NAME_ALIASES = {
    "LOX": "Oxygen",
    "GOX": "Oxygen",
    "GN2": "Nitrogen",
    "NOX": "NitrousOxide"
}

PA_PER_PSIA = 6894.757293168
KG_PER_M3_PER_G_PER_ML = 1000.0
PA_S_PER_POISE_DYNAMIC_VISCOSITY = 0.1
J_PER_KG_K_PER_BTU_PER_LBM_R = 4186.80058485
W_PER_M_K_PER_BTU_PER_HR_FT_R = 1.730734666



def normalize_fluid_name(name: str) -> str:
    return name.upper().replace("-", "").replace("_", "").replace(" ", "")



class CoolPropBackend:

    def __init__(self, name: str, pressure: float, input_name: str, input_value: float):
        self.name = name
        self.pressure = pressure
        self.input_name = input_name
        self.input_value = input_value

    def get(self, property_name: str) -> float:
        return PropsSI(property_name, "P", self.pressure, self.input_name, self.input_value, self.name)




class RocketPropsBackend:

    SUPPORTED_PROPERTIES = {"T", "Dmass", "Cpmass", "V", "L"}

    def __init__(self, name: str, pressure: float, temperature: float):
        self.fluid = get_prop(name)
        self.pressure = pressure
        self.temperature = temperature

    def get(self, property_name: str) -> float:
        if property_name not in self.SUPPORTED_PROPERTIES:
            raise NotImplementedError(f"RocketProps does not support '{property_name}' for a pressure-temperature state.")

        temperature_deg_r = self.temperature * 9.0 / 5.0
        pressure_psia = self.pressure / PA_PER_PSIA

        if property_name == "T":
            return self.temperature
        if property_name == "Dmass":
            return self.fluid.SG_compressed(temperature_deg_r, pressure_psia) * KG_PER_M3_PER_G_PER_ML
        if property_name == "Cpmass":
            return self.fluid.CpAtTdegR(temperature_deg_r) * J_PER_KG_K_PER_BTU_PER_LBM_R
        if property_name == "V":
            return self.fluid.Visc_compressed(temperature_deg_r, pressure_psia) * PA_S_PER_POISE_DYNAMIC_VISCOSITY
        return self.fluid.CondAtTdegR(temperature_deg_r) * W_PER_M_K_PER_BTU_PER_HR_FT_R




class Fluid:

    def __init__(self, name: str, pressure: float, temperature: float = None, enthalpy: float = None):
        if (temperature is None) == (enthalpy is None):
            raise ValueError("Provide either temperature or enthalpy, but not both.")

        self.name = name
        self._pressure = float(pressure)
        normalized_name = normalize_fluid_name(name)
        backend_name = FLUID_NAME_ALIASES.get(normalized_name, name)

        if normalized_name in ROCKETPROPS_FLUIDS:
            if enthalpy is not None:
                raise ValueError(f"{name} does not support pressure-enthalpy flashes.")
            self._backend = RocketPropsBackend(normalized_name, self._pressure, float(temperature))
        else:
            input_name = "T" if temperature is not None else "Hmass"
            input_value = temperature if temperature is not None else enthalpy
            self._backend = CoolPropBackend(backend_name, self._pressure, input_name, float(input_value))

    def _get(self, property_name: str) -> float:
        return self._backend.get(property_name)

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
    def dynamic_viscosity(self) -> float:
        """Dynamic viscosity in Pa*s."""
        return self._get("V")

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity in m^2/s."""
        return self.dynamic_viscosity / self.density

    @property
    def thermal_conductivity(self) -> float:
        return self._get("L")

    @property
    def speed_of_sound(self) -> float:
        return self._get("A")

    @property
    def quality(self) -> float:
        return self._get("Q")
