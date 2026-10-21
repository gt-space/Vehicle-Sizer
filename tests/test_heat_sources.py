from copy import deepcopy
from types import MappingProxyType, SimpleNamespace

import pytest

from diagnostics.errors import TrialDomainError
from Fluids.design import prepare_design_inputs
from Fluids.FluidState import FluidState, NodeState
from Thermals.heat_sources import build_heat_source, evaluate_heat, thermal_model


def state(rho, phase='gas'):
    return NodeState(fluids={'nitrogen': FluidState('nitrogen', phase, {'rho': rho})})


def test_prescribed_rates_do_not_depend_on_trial_density():
    config = {'model': 'Prescribed', 'heat_rate': {'gas': 100.}}
    source = build_heat_source(config)
    first = state(200.)
    before = deepcopy(first)
    assert evaluate_heat(source, first, 0.) == {'gas': 100.}
    assert evaluate_heat(source, state(100.), 10.) == {'gas': 100.}
    assert first == before
    source['gas'] = 200.
    assert config['heat_rate']['gas'] == 100.


def test_prescribed_time_callback():
    source = build_heat_source({'model': 'Prescribed', 'heat_rate': lambda t: {'gas': 3*t}})
    assert evaluate_heat(source, state(2.), 4.) == {'gas': 12.}


@pytest.mark.parametrize('config', [None, {}, {'model': None}, {'model': 'None'}])
def test_adiabatic_default(config):
    assert evaluate_heat(build_heat_source(config), state(1.), 0.) == {}


def test_external_thermal_rates_and_time_callback_share_interface():
    trial = state(2.)
    prescribed = MappingProxyType({'gas': 12.})
    output = evaluate_heat(prescribed, trial, 0.)
    assert output == {'gas': 12.}
    output['gas'] = -1.
    assert prescribed['gas'] == 12.
    assert evaluate_heat(lambda t: {'gas': t * 3}, trial, 4.) == {'gas': 12.}
    with pytest.raises(ValueError, match='absent phase'):
        evaluate_heat({'liquid': 1.}, trial, 0.)
    with pytest.raises(ValueError, match='wall thermal solve'):
        evaluate_heat(build_heat_source({'model': 'Aeroheating'}), trial, 0.)


@pytest.mark.parametrize('config', [
    {'model': 'wrong'}, {'model': 'None', 'insulated': True},
    {'model': 'Aeroheating', 'insulated': True}, False,
    {'model': 'DensityPowerLaw'},
    {'model': 'Prescribed'}, {'model': 'Prescribed', 'heat_rate': 100.},
    {'model': 'Prescribed', 'heat_rate': {'gas': float('inf')}},
    {'model': 'Prescribed', 'heat_rate': {'solid': 10.}},
    {'model': 'Prescribed', 'heat_rate': {}, 'area': 1.},
])
def test_invalid_config_rejected(config):
    with pytest.raises(ValueError):
        build_heat_source(config)


def test_phase_selection_and_finite_output():
    source = build_heat_source({'model': 'Prescribed', 'heat_rate': {'liquid': 10.}})
    assert evaluate_heat(source, state(200., 'liquid'), 0.) == {'liquid': 10.}
    with pytest.raises(ValueError, match='absent phase'):
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
                     'copv': {'thermal': {'model': 'Prescribed', 'heat_rate': {'gas': 100.}}}}}
    network = ThermalNetwork(cfg, SimpleNamespace(sections=[fuel, ox, copv],
        fluid_sections=tuple((s.tank_id, s, None) for s in (fuel, ox, copv))))
    assert set(network.nodes) == {'fuel', 'ox'}


def test_real_config_binds_prescribed_and_default_sources():
    from Configs.loader import load_config
    from FluidTables.PropertyModels import property_sources
    from Vehicle.Vehicle import Vehicle
    from Fluids.PropSystem import PropSystem
    cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
    cfg['tanks']['press_tank']['thermal'] = {'model': 'Prescribed', 'heat_rate': {'gas': 100.}}
    pure, combustion = property_sources(cfg)
    template, states, wall_pressures = prepare_design_inputs(cfg)
    vehicle = Vehicle(cfg, pure, initial_states=states, wall_pressures=wall_pressures)
    with PropSystem(cfg, vehicle.tanks, pure, combustion, initial_states=states, template=template) as prop:
        nodes = {definition['tank_id']: key for key, definition in prop.node_definitions.items()
                 if 'tank_id' in definition}
        source = prop.network.heat_sources[nodes['press_tank']]
        assert evaluate_heat(source, state(2.), 0.) == {'gas': 100.}
        for tank in ('ox_tank', 'fuel_tank'):
            assert prop.network.heat_sources[nodes[tank]].evaluate(state(1.)) == {}


@pytest.mark.parametrize('kind', ['gas', 'propellant'])
def test_prescribed_heat_increases_sealed_tank_energy(kind):
    from Fluids import FluidNetwork
    from test_fluid_network import gas_tank, wet_tank, Properties
    tank = gas_tank() if kind == 'gas' else wet_tank()
    rates = {'gas': 100.} if kind == 'gas' else {'gas': 100., 'liquid': 200.}
    tank['thermal'] = {'model': 'Prescribed', 'heat_rate': rates}
    with FluidNetwork({'tank': tank}, {}, fluid_properties=Properties()) as network:
        network.initialize()
        def energy():
            values = network.state.nodes['tank']
            return values['U'] if kind == 'gas' else values['U_liq'] + values['U_ull']
        initial = energy()
        network.update(.1)
        assert energy() - initial == pytest.approx(sum(rates.values()) * .1, abs=1e-4)
