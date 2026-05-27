from rocketcea.cea_obj_w_units import CEA_Obj

class Engine:

    def __init__(self, cfg: dict):

        self.cfg = cfg
        self.mass = self.cfg["engine"]["mass"]
        self.chamber_pressure = self.cfg["engine"]["chamber_pressure"]
        self.exit_pressure = self.cfg["engine"]["exit_pressure"]
        self.mixture_ratio = self.cfg["engine"]["mixture_ratio"]
        self.perfectly_expanded_thrust = self.cfg["engine"]["perfectly_expanded_thrust"]
        self.cea = CEA_Obj(
            oxName=self.oxidizer,
            fuelName=self.fuel,
            pressure_units="Pa",
            cstar_units="m/s",
            isp_units="sec",
            temperature_units="K",
            sonic_velocity_units="m/s",
            density_units="kg/m^3",
            specific_heat_units="J/kg-K",
        )

    def build(self):
        self._get_cstar()
        self._get_expansion_ratio()
        self._get_Cf()
        self._get_areas()

    def _get_cstar(self):
        self.cstar = self.cea.get_Cstar(self.chamber_pressure, self.mixture_ratio) * self.cfg["engine"]["cstar_efficiency"]

    def _get_expansion_ratio(self):
        PcOvPe = self.chamber_pressure / self.exit_pressure
        self.eps = self.cea.get_eps_at_PcOvPe(self.chamber_pressure, self.mixture_ratio, PcOvPe)

    def _get_Cf(self):
        Cf_ideal = self.cea.getFrozen_PambCf(self.exit_pressure, self.chamber_pressure, self.mixture_ratio, self.eps, 1)
        self.Cf = Cf_ideal[1] * self.cfg["engine"]["cf_efficiency"]

    def _get_areas(self):
        self.throat_area = self.perfectly_expanded_thrust / (self.chamber_pressure * self.Cf)
        self.exit_area = self.throat_area * self._get_expansion_ratio()