from rocketcea.cea_obj import CEA_Obj

class Engine:

    def __init__(self, cfg):

        self.cfg = cfg
        self.mass = None
        self.EI = None

    def _get_