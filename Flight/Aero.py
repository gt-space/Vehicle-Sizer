import numpy as np

class Aero:

    def __init__(self, cfg, vehicle):

        self.cfg = cfg
        self.vehicle = vehicle

    def _distribute(self, C_tot: float, lat_area: np.ndarray) -> np.ndarray:

        w = lat_area / np.sum(lat_area)
        return C_tot * w
        
    def _get_nose_CNa(self, M: float) -> float:

        P = self._get_comp_factor(M)
        return 2 * P
    
    def _get_body_CNa(self, M: float, alpha: float) -> float:

        L = sum(sec.length for sec in body_sections)
        A_plan = self.D * L
        P = self._get_comp_factor(M)
        K = 1.1
        return P * K * (A_plan / self.A_ref) * (np.sin(alpha)**2 / alpha)
    
    def _get_tail_CNa(self, M: float, alpha: float) -> float:

        A_exit = 0.25 * np.pi * tail.exit_diameter**2
        P = self._get_comp_factor(M)
        return P * (2 / self.A_ref) * (A_exit - self.A_ref) * (np.sin(alpha) / alpha)
    
    def _get_fin_CNa(self, M: float) -> float:

        P = self._get_comp_factor(M)
        CNa1 = 
        Kfb = 

        return P * (N * 0.5) * Kfb * CNa1

    def _get_comp_factor(self, M: float) -> float:

        return 1 / self._get_beta(M)

    @staticmethod
    def _get_beta(M: float) -> float:

        if M <= 0.8:
            return np.sqrt(1 - M**2)

        elif M < np.sqrt(34 / 5):
            return np.sqrt(1 - 0.8**2)

        else:
            return np.sqrt(M**2 - 1)