from CoolProp.CoolProp import PropsSI
import PropTank

class OxTank(PropTank):

    def __init__(self, cfg: dict):
        
        super().__init__(cfg)

    def _get_volume(self):

        T = 100.0
        p = self.pressure
        rho = PropsSI('D', 'T', T, 'P', p, 'Oxygen')
        m = self.mass

        V = self.cfg["ox_tank"]["ullage_factor"] * m / rho

        return V