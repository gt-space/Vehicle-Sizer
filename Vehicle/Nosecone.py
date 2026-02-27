import matproplib as mp
from Vehicle.Section import Section

class Nosecone(Section):

    def __init__(self, cfg: dict):
        
        super().__init__(cfg)
        self.n = int(np.ceil(self.length / self.dx))

    def get_mass(self):

    def get_EI(self):

    def get_lat_area(self):