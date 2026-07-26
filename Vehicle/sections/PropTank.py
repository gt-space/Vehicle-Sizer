import numpy as np
import matproplib as mp
from CoolProp.CoolProp import PropsSI
from .Section import Section
from ..utils import distribute as dist
from ..utils import aero
from ..utils import geometry as geo
from ..utils import heating

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
        self.P_liq0 = P_liq0
        self.T_liq0 = T_liq0
        self.gas = cfg["press_tank"]["pressurant"]
        self.TankVolume = self._tank_volume()
        self._get_length()
        self.n = int(np.ceil(self.length / self.dx))

    def get_mass(self):
        dry_mass = self._get_dry_mass()
        self.dry_mass = dist.uniform(dry_mass, self.n)
        self.mass = self.dry_mass

    def _get_dry_mass(self) -> float:
        D = self.OMLD
        D_pass = self.passthrough_diameter
        self._get_length()
        t = self.wall_thickness
        t_pass = 0.00254
        e = self.ellipse_ratio
        rho = mp.db.get_material(self.material).get("density")

        k = (
            2 * e
            + (1 / np.sqrt(e**2 - 1))
            * np.log((e + np.sqrt(e**2 - 1)) / (e - np.sqrt(e**2 - 1)))
        )

        V_end = ((1/4) * np.pi * (D - 2*t) * t * k) / (2 * e)
        V_cyl = geo.annulus_volume(D * 0.5, D * 0.5 - t, self.cyl_length)
        V_pass = geo.annulus_volume(D_pass * 0.5, D_pass * 0.5 - t_pass, self.cyl_length)

        return rho * (V_end + V_cyl + V_pass)

    def _get_pressure(self):
        return 1e6

    def _tank_volume(self) -> float:
        return self._liquid_capacity() * self.ullage_factor

    def _liquid_capacity(self) -> float:
        return self.prop_mass / self.get_liquid_density(self.T_liq0, self.P_liq0)

    def get_ullage_volume(self, T_liq: float, P_liq: float, mOX: float) -> float:
        return self.TankVolume - (mOX / self.get_liquid_density(T_liq, P_liq))

    def get_liquid_density(self, T_liq: float, P_liq: float) -> float:
        return PropsSI("D", "T", T_liq, "P", P_liq, self.medium)

    def _get_wall_thickness(self):
        sigma = mp.db.get_material(self.material).get("yield_strength", 400.0)
        self.pressure = self._get_pressure()
        t = 1.4 * (self.pressure * self.OMLD) / (2 * sigma)
        return max(t, 1/16 * 0.0254)

    def _get_length(self):
        D = self.OMLD
        D_pass = self.passthrough_diameter
        self.volume = self._tank_volume()
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
        r_i = r_o - self._get_wall_thickness()
        E = mp.db.get_material(self.material).get("elastic_modulus", 300.0)
        self.EI = dist.uniform_full(E * geo.annulus_second_moment(r_o, r_i), self.n)

    def get_area(self):
        r = self.OMLD * 0.5
        self.lat_area = dist.uniform(geo.cylinder_lateral_area(r, self.length), self.n)
        self.surf_area = dist.uniform(geo.cylinder_surface_area(r, self.length), self.n)

    def get_MOI(self):
        r = self.cfg["vehicle"]["OMLD"] * 0.5
        self.cg = np.sum(self.mass * self.station) / np.sum(self.mass)
        self.Ixx = np.sum(self.mass * r**2)
        self.Iyy = np.sum(self.mass * (self.station - self.cg)**2)

    def get_CNa(self, M: float, alpha: float):
        A_plan = self.cfg["vehicle"]["OMLD"] * self.length
        self.CNa = dist.weighted(aero.body_CNa(M, alpha, A_plan, self.ref_area), self.lat_area)

    def drain_prop(self, dm: float):
        self.prop_mass = max(self.prop_mass - dm, 0.0)

        self.prop_end_station = self.end_station
        self.prop_start_station = self.start_station - self.prop_height

        T = 300
        p = 1e6
        rho = PropsSI("D", "T", T, "P", p, self.medium)
        dV = dm / rho
        self.ullage_volume += dV

        rho_press = 1600
        self.press_mass = self.ullage_volume * rho_press

    def get_heat_flux(self, ):
        self.heat_flux = heating.get_body_heating()