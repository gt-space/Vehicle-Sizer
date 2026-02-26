from CoolProp.CoolProp import PropsSI
import Vehicle.PropTank as PropTank

class FuelTank(PropTank): # potentially make child class of PropTank instead of having double abstract classes

    def __init__(self, cfg: dict):
        
        super().__init__(cfg)

    def _get_volume(self):

        T = 300.0
        p = self.pressure
        rho = PropsSI('D', 'T', T, 'P', p, 'n-Dodecane')
        m = self.mass

        V = self.cfg["fuel_tank"]["ullage_factor"] * m / rho

        return V