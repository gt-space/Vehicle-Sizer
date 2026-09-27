"""Small physical fixtures for propulsion sizing and template tests."""
from dataclasses import dataclass
from CoolProp.CoolProp import PropsSI


@dataclass(frozen=True)
class GasGeometry:
    volume: float
    internal_area: float = 1.0

    @staticmethod
    def axial_mass(mass):
        return [mass]

@dataclass(frozen=True)
class TankGeometry:
    volume: float
    internal_area: float = 2.0

    def fill_state(self, liquid_volume):
        fraction = liquid_volume / self.volume
        return {
            "fill_height": fraction,
            "liquid_contact_area": self.internal_area * fraction,
            "ullage_contact_area": self.internal_area * (1.0 - fraction),
        }

    @staticmethod
    def axial_mass(liquid_volume, liquid_mass, ullage_mass):
        return [liquid_mass + ullage_mass]

class GeometrySource:
    def __init__(self, geometry, prop_mass=None):
        self.geometry = geometry
        if prop_mass is not None:
            self.prop_mass = prop_mass

    def get_fluid_geometry(self):
        return self.geometry

class FakeCEA:
    @staticmethod
    def get_eps_at_PcOvPe(Pc, MR, PcOvPe):
        return 5.0

    @staticmethod
    def get_Cstar(Pc, MR):
        return 1500.0

    @staticmethod
    def getFrozen_PambCf(Pamb, Pc, MR, eps, frozen):
        return 0.0, 1.5

    @staticmethod
    def get_Chamber_MolWt_gamma(Pc, MR, eps):
        return 24.0, 1.2

    @staticmethod
    def get_Temperatures(Pc, MR, eps):
        return (3000.0,)

    @staticmethod
    def get_Chamber_H(Pc, MR, eps):
        return 5.0e6

def config():
    return {
        "tanks": {"press_tank": {"type": "pressurant", "design_pressure": 30e6}},
        "prop_system": {
            "template": "Configs/templates/pressure_fed_bang_bang.yaml",
            "Pc_target": 2.0e6,
            "MR_target": 3.0,
            "thrust_target": 12_000.0,
            "fuel_inj_stiffness": 0.2,
            "ox_inj_stiffness": 0.2,
            "fuel_tank_inj_dp": 200_000.0,
            "ox_tank_inj_dp": 200_000.0,
            "expansion_ratio": 5.0,
            "nozzle_cd": 1.0,
            "bang_bang": {
                branch_id: {
                    "duty_cycle": 0.5,
                    "collapse_factor": 1.2,
                    "min_temperature": 220.0,
                    "pressure_band": 68_947.6,
                    "initially_open": True,
                }
                for branch_id in ("OX_BANGBANG", "FUEL_BANGBANG")
            },
            "initial_conditions": {
                "press_tank": {
                    "fluid": "Nitrogen",
                    "T": 300.0,
                },
                "ox_tank": {
                    "fluid": "Oxygen",
                    "gas_fluid": "Nitrogen",
                    "P": 2.6e6,
                    "T": 100.0,
                    "gas_T": 300.0,
                },
                "fuel_tank": {
                    "fluid": "n-Dodecane",
                    "gas_fluid": "Nitrogen",
                    "P": 2.6e6,
                    "T": 300.0,
                    "gas_T": 300.0,
                },
            },
        },
        "engine": {
            "property_source": "cea",
            "oxidizer": "LOX",
            "fuel": "RP-1",
            "cstar_efficiency": 1.0,
            "cf_efficiency": 1.0,
            "exit_pressure": 100_000.0,
        },
    }

def tanks(tank_pressure=2.6e6):
    ox_density = PropsSI("Dmass", "P", tank_pressure, "T", 100.0, "Oxygen")
    fuel_density = PropsSI(
        "Dmass", "P", tank_pressure, "T", 300.0, "n-Dodecane"
    )
    return (
        GeometrySource(TankGeometry(90.0 / ox_density * 1.1), prop_mass=90.),
        GeometrySource(TankGeometry(30.0 / fuel_density * 1.1), prop_mass=30.),
        GeometrySource(GasGeometry(0.05)),
    )
