import numpy as np
from Configs.loader import load_config
from FluidTables.PropertyModels import CoolPropPropertySource
from Vehicle.Vehicle import Vehicle
from Vehicle.Engine import Engine
from Thermals.ThermalNetwork import ThermalNetwork

def test_passthrough_zero_contact_cells_keep_positive_thermal_mass():
    cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
    cfg['vehicle']['dx'] = .001
    vehicle = Vehicle(cfg, CoolPropPropertySource())
    vehicle.engine = Engine(30, .5715, .01)
    vehicle.sections = vehicle._build_sections()
    vehicle._stack_sections()
    fuel = vehicle.tanks['fuel_tank']
    assert np.any(fuel.get_thermal_internal_area() == 0)
    mass = fuel.get_thermal_shell_mass()
    assert np.all(mass > 0)
    np.testing.assert_allclose(mass.sum(), fuel.shell_mass.sum())
    network = ThermalNetwork(cfg, vehicle, selections={
        'fuel_tank': {'model': 'Aeroheating'}, 'ox_tank': {'model': 'Aeroheating'}})
    assert set(network.nodes) == {'fuel_tank', 'ox_tank'}
