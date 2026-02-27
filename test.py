import numpy as np
from Vehicle.InterTank import InterTank
from Configs.loader import load_config

cfg = load_config("Configs/kerolox_pumped.yaml")
L = 0.12
P = 6000.0
M = 4000.0

intertank = InterTank(cfg, L, P, M)
intertank.get_mass()
intertank.get_EI()
intertank.get_lat_area()

print("EI:", intertank.EI[0])