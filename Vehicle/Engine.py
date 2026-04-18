from rocketcea.cea_obj import CEA_Obj

class Engine:

    def __init__(self, cfg: dict):

        self.cfg = cfg
        self.cea = CEA_Obj(ox=self.oxidizer, fuel=self.fuel)
        self.mass = self.cfg["engine"]["mass"]

    def _get_

    def _get_cstar(self):

        cstar_ideal = self.cea.get_Cstar()
        self.cstar = cstar_ideal * self.cfg["engine"]["cstar_efficiency"]

    def _get_Cf(self):

        Cf_ideal = self.cea.getFrozen_PambCf()
        self.Cf = Cf_ideal * self.cfg["engine"]["cf_efficiency"]

    def _get_areas(self):

        self.throat_area = 

        self._get_expansion_ratio()
        self.exit_area = self.throat_area * self.eps

    def _get_expansion_ratio(self):

        self.eps = self.cea.get_eps_at_PcOvPe()