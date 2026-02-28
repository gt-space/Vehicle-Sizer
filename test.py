import numpy as np
from Vehicle.InterTank import InterTank
from Vehicle.PressTank import PressTank
from Configs.loader import load_config

cfg = load_config("Configs/kerolox_pumped.yaml")

L1 = 0.12
P1 = 6000.0
M1 = 10000.0

s1 = InterTank(cfg, L1, P1, M1)
s1.get_mass()
s1.get_EI()
s1.get_lat_area()

s2 = PressTank(cfg)
s2.get_mass()
s2.get_EI()
s2.get_lat_area()

L3 = 0.36
P3 = 15000.0
M3 = 5000.0

s3 = InterTank(cfg, L3, P3, M3)
s3.get_mass()
s3.get_EI()
s3.get_lat_area()

sections = [s1, s2, s3]
mass_list = [sec.mass for sec in sections]
mass = np.concatenate(mass_list)

print("Mass1:", np.sum(s1.mass))
print("EI1:", s1.EI[0])

print("Mass2:", np.sum(s2.mass))
print("EI2:", s2.EI[0])

print("Mass3:", np.sum(s3.mass))
print("EI3:", s3.EI[0])

print("MassTot:", np.sum(mass))