import matproplib as mp

cf = mp.db.get_material("carbon_fiber_standard")
rho = cf.get("density")
print(rho)