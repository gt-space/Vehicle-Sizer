import numpy as np

class IncompressibleFlow:

    @staticmethod
    def mass_flow_from_cda(P1, P2, rho, CdA):
        return np.sign(P1-P2) * CdA * np.sqrt(2*rho*np.abs(P1-P2))

    @staticmethod
    def cda_from_mass_flow(P1, P2, rho, mass_flow):
        return mass_flow / (np.sign(P1-P2) * np.sqrt(2*rho*np.abs(P1-P2)))
