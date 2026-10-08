from copy import deepcopy
from types import SimpleNamespace

import pytest

from diagnostics.errors import TrialDomainError
from Fluids.FluidState import FluidState, NodeState
from Thermals.heat_sources import build_heat_source, evaluate_heat, thermal_model


def state(rho, phase='gas'):
    return NodeState(fluids={'nitrogen': FluidState('nitrogen', phase, {'rho': rho})})


def test_exact_correlation_and_trial_density():
    source = build_heat_source({'model': 'DensityPowerLaw', 'reference_area': 1.},
                               SimpleNamespace(internal_area=2.))
    first = state(209.559)
    before = deepcopy(first)
    assert evaluate_heat(source, first, 0.) == {'gas': pytest.approx(31746.6)}
    assert evaluate_heat(source, state(104.7795), 10.)['gas'] == pytest.approx(31746.6 * .5**1.2609)
    assert first == before
    assert evaluate_heat(source, first, 10.) == evaluate_heat(source, first, 0.)
    assert deepcopy(source) == source


@pytest.mark.parametrize('config', [None, {}, {'model': None}, {'model': 'None'}])
def test_adiabatic_default(config):
    assert evaluate_heat(build_heat_source(config), state(1.), 0.) == {}


def test_external_thermal_rates_and_time_callback_share_interface():
    trial = state(2.)
    assert evaluate_heat({'gas': 12.}, trial, 0.) == {'gas': 12.}
    assert evaluate_heat(lambda t: {'gas': t * 3}, trial, 4.) == {'gas': 12.}
    with pytest.raises(ValueError, match='wall thermal solve'):
        evaluate_heat(build_heat_source({'model': 'Aeroheating'}), trial, 0.)


@pytest.mark.parametrize('config', [
    {'model': 'wrong'}, {'model': 'None', 'insulated': True},
    {'model': 'Aeroheating', 'insulated': True}, False,
    {'model': 'DensityPowerLaw'},
    {'model': 'DensityPowerLaw', 'area': 1., 'reference_area': 0.},
    {'model': 'DensityPowerLaw', 'area': -1., 'reference_area': 1.},
])
def test_invalid_config_rejected(config):
    with pytest.raises(ValueError):
        build_heat_source(config)


@pytest.mark.parametrize('rho', [0., -1., float('nan'), float('inf')])
def test_invalid_trial_density_is_recoverable(rho):
    source = build_heat_source({'model': 'DensityPowerLaw', 'area': 1., 'reference_area': 1.})
    with pytest.raises(TrialDomainError):
        evaluate_heat(source, state(rho), 0.)


def test_phase_selection_and_finite_output():
    source = build_heat_source({'model': 'DensityPowerLaw', 'area': 1., 'reference_area': 1., 'phase': 'liquid'})
    assert evaluate_heat(source, state(209.559, 'liquid'), 0.) == {'liquid': pytest.approx(15873.3)}
    with pytest.raises(ValueError, match='matching'):
        evaluate_heat(source, state(2.), 0.)
    with pytest.raises(TrialDomainError):
        evaluate_heat({'gas': float('inf')}, state(2.), 0.)


def test_mixed_wall_selection():
    from Thermals import ThermalNetwork
    from test_thermal_network import WetSection
    fuel, ox, copv = WetSection(), WetSection(), WetSection()
    fuel.tank_id, ox.tank_id, copv.tank_id = 'fuel', 'ox', 'copv'
    cfg = {'thermal': {'initial_temperature': 300., 'sink_temperature': 288.},
           'tanks': {'fuel': {'thermal': {'model': 'Aeroheating'}},
                     'ox': {'thermal': {'model': 'Aeroheating'}},
                     'copv': {'thermal': {'model': 'DensityPowerLaw', 'reference_area': 1.}}}}
    network = ThermalNetwork(cfg, SimpleNamespace(sections=[fuel, ox, copv]))
    assert set(network.nodes) == {'fuel', 'ox'}


def test_real_config_binds_area_and_defaults():
    from Configs.loader import load_config
    from simulation import property_sources
    from Vehicle.Vehicle import Vehicle
    from Fluids.PropSystem import PropSystem
    cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
    pure, combustion = property_sources(cfg)
    vehicle = Vehicle(cfg, pure)
    with PropSystem(cfg, vehicle.tanks, pure, combustion) as prop:
        nodes = {definition['tank_id']: key for key, definition in prop.node_definitions.items()
                 if 'tank_id' in definition}
        source = prop.network.heat_sources[nodes['press_tank']]
        assert source.area == pytest.approx(vehicle.tanks['press_tank'].get_fluid_geometry().internal_area)
        assert source.reference_area == cfg['tanks']['press_tank']['thermal']['reference_area']
        for tank in ('ox_tank', 'fuel_tank'):
            assert prop.network.heat_sources[nodes[tank]].evaluate(state(1.)) == {}
