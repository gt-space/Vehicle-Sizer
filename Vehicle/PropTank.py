import numpy as np
import matproplib as mp
from CoolProp.CoolProp import PropsSI
from Vehicle.Section import Section
from Flight import PropSystem

class PropTank(Section):

    def __init__(self, cfg: dict, medium: str, prop_mass: float, material: str, passthrough_diameter: float, ellipse_ratio: float, ullage_factor: float, P_liq0: float, T_liq0: float):
        
        super().__init__(cfg)
        self.passthrough_diameter = passthrough_diameter
        self.ellipse_ratio = ellipse_ratio
        self.ullage_factor = ullage_factor
        self.OMLD = cfg["vehicle"]["OMLD"]
        self.prop_mass = prop_mass
        self.material = material
        self.medium = medium
        self.P_liq0 = P_liq0 # Initial ullage pressure
        self.T_liq0 = T_liq0 # Initial ullage temperature
        self.gas = cfg.get("pressurant")
        self.TankVolume = self._tank_volume(self) 

    def get_mass(self):

        dry_mass = self._get_dry_mass()
        self.n = int(np.ceil(self.length / self.dx))
        self.dry_mass = np.full(self.n, dry_mass / self.n)
        self.mass = self.dry_mass

    def _get_dry_mass(self) -> float:

        D = self.OMLD
        D_pass = self.passthrough_diameter
        self._get_length()
        t = self.wall_thickness
        t_pass = 0.00254
        e = self.ellipse_ratio

        mat = mp.db.get_material(self.material)
        rho = mat.get("density")

        k = (
            2 * e
            + (1 / np.sqrt(e**2 - 1))
            * np.log(
                (e + np.sqrt(e**2 - 1)) /
                (e - np.sqrt(e**2 - 1))
            )
        )

        V_end = (
            (1/4) * np.pi * (D - 2*t) * t * k
        ) / (2 * e)

        V_cyl = (
            (1/4) * np.pi * self.cyl_length
            * (D**2 - (D - 2*t)**2)
        )

        V_pass = (
            (1/4) * np.pi * self.cyl_length
            * (D_pass**2 - (D_pass - 2*t_pass)**2)
        )

        V = V_end + V_cyl + V_pass
        m = rho * V

        return m

    def _get_pressure(self):
        
        return 1e6

    def _tank_volume(self) -> float:
        # Sized using initial ullage pressure and temperature 
        return self._liquid_capacity() * self.ullage_factor

    def _liquid_capacity(self) -> float:
        # Required liquid volume of tank, not to be confused with dynamic volume tracking
        rho = self.get_liquid_density(self.T_liq0, self.P_liq0)
        return self.prop_mass / rho

    def get_ullage_volume(self, T_liq: float, P_liq: float, mOX: float) -> float:
        return self.TankVolume - (mOX / self.get_liquid_density(T_liq, P_liq))
    
    def get_liquid_density(self, T_liq: float, P_liq: float) -> float:
        return PropsSI("D", "T", T_liq, "P", P_liq, self.medium)

    def _get_wall_thickness(self):

        mat = mp.db.get_material(self.material)
        T = 400.0
        sigma = mat.get("yield_strength", T)
        FOS = 1.4

        self.pressure = self._get_pressure()
        t = FOS * (self.pressure * self.OMLD) / (2 * sigma)
        t_min = 1/16 * 0.0254
        t = max(t, t_min)

        return t
    
    def _get_length(self):

        D = self.OMLD
        D_pass = self.passthrough_diameter
        self.get_tank_volume()
        self.wall_thickness = self._get_wall_thickness()

        V_end = (np.pi / (12 * self.ellipse_ratio)) * (D - (2 * self.wall_thickness))**3

        numerator = (
            self.volume
            - 2 * V_end
            + (4 * np.pi / self.ellipse_ratio) * D_pass**2 * D
        )

        denominator = (
            (np.pi / 4)
            * ((D - 2*self.wall_thickness)**2 - D_pass**2)
        )

        self.cyl_length = numerator / denominator
        self.length = self.cyl_length + (D / self.ellipse_ratio)

    def get_EI(self):

        r_o = self.OMLD * 0.5

        t = self._get_wall_thickness()
        r_i = r_o - t

        I = np.pi * 0.25 * (r_o**4 - r_i**4)
        mat = mp.db.get_material(self.material)
        T = 300.0
        E = mat.get("elastic_modulus", T)

        EI = E * I
        self.EI = np.full(self.n, EI)

    def get_area(self):

        D = self.OMLD
        self.lat_area = np.full(self.n, D * self.dx)
        self.surf_area = self.lat_area * np.pi

    def get_CNa(self, M: float, alpha: float):

        K = 1.1
        P = self.get_comp_factor(M)
        A_plan = self.length * self.cfg["vehicle"]["OMLD"]
        CNa = K * P * (A_plan / self.ref_area) * (np.sin(alpha)**2 / alpha)

        self.CNa = self.distribute(CNa, self.lat_area)

    def drain_prop(self, dm: float):

        self.prop_mass = max(self.prop_mass - dm, 0.0)

        fill_ratio = self.prop_mass / self.prop_mass_initial
        self.prop_end_station = self.end_station
        self.prop_start_station = self.start_station - self.prop_height

        T = 300
        p = 1e6
        rho = PropsSI("D", "T", T, "P", p, self.medium)
        dV = dm / rho
        self.ullage_volume += dV

        rho_press = 1600
        self.press_mass = self.ullage_volume * rho_press