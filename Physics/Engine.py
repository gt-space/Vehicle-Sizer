from rocketcea.cea_obj_w_units import CEA_Obj


kerolox_engine = CEA_Obj(oxName='LOX', fuelName='RP-1',
                    temperature_units='degK', cstar_units='m/sec',
                    specific_heat_units='kJ/kg degK',
                    sonic_velocity_units='m/s', enthalpy_units='J/kg',
                    density_units='kg/m^3', pressure_units='Pa')


def get_cstar_ideal(Pc, MR, cea_obj: CEA_Obj = kerolox_engine):
    """Returns ideal characteristic velocity (m/s)"""
    return cea_obj.get_Cstar(Pc, MR)


def get_cf_ideal(Pc, MR, Pamb, eps, nfz=0, cea_obj: CEA_Obj = kerolox_engine):
    """Returns ideal thrust coefficient"""
    if nfz==2:
        _, cf, _ = cea_obj.getFrozen_PambCf(Pamb, Pc, MR, eps, 1)
    elif nfz==1:
        _, cf, _ = cea_obj.getFrozen_PambCf(Pamb, Pc, MR, eps, 0)
    else:
        _, cf, _ = cea_obj.get_PambCf(Pamb, Pc, MR, eps)
    return cf


def get_eps(Pc, MR, Pe, nfz=0, cea_obj: CEA_Obj = kerolox_engine):
    """Returns expansion ratio"""
    if nfz==2:
        return cea_obj.get_eps_at_PcOvPe(Pc, MR, Pc/Pe, 1, 1)
    elif nfz==1:
        return cea_obj.get_eps_at_PcOvPe(Pc, MR, Pc/Pe, 1, 0)
    else:
        return cea_obj.get_eps_at_PcOvPe(Pc, MR, Pc/Pe, 0, 0)


def get_chamber_density(Pc, MR, cea_obj: CEA_Obj = kerolox_engine):
    """Returns chamber gas density (kg/m^3)"""
    return cea_obj.get_Chamber_Density(Pc, MR)
