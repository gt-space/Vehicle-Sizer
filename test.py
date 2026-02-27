import numpy as np
from Vehicle.InterTank import InterTank
from Configs.loader import load_config

cfg = load_config("Configs/kerolox_pumped.yaml")
L = 0.13
P = 7000.0
M = 7000.0

intertank = InterTank(cfg, L, P, M)
intertank.get_mass()
intertank.get_EI()
intertank.get_lat_area()

print("Mass:", np.sum(intertank.mass))
print("EI:", intertank.EI[0])
print("EI_clamshell:", intertank._get_clamshell_EI())
print("EI_stringers:", intertank._get_stringer_EI(intertank.stringer_thickness))
print("a_stringer:", intertank.stringer_thickness)