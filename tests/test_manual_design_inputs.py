"""Supplied geometry replaces sizing; mass and COM still follow the cell model."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from AeroTables import DragModel
from Configs.loader import load_config
from FluidTables.PropertyModels import CEAPropertySource
from Fluids.PropSystem import PropSystem
from Vehicle.Engine import Engine
from Vehicle.Vehicle import Vehicle
from simulation import property_sources
from test_prop_system import config, build


@pytest.fixture
def vespula():
    cfg = load_config('Configs/Vespula.yaml')
    pure, combustion = property_sources(cfg)
    vehicle = Vehicle(cfg, pure)
    with PropSystem(cfg, vehicle.tanks, pure, combustion) as prop:
        vehicle.build(Engine(cfg['engine']['mass'], cfg['engine']['length'], prop.exit_area))
        yield cfg, vehicle, prop


def test_explicit_engine_inputs_skip_sizing_and_are_not_changed_by_targets(monkeypatch):
    cfg = config()
    cfg['engine'].update(throat_area=.0039, expansion_ratio=4.066)
    del cfg['engine']['exit_pressure']
    del cfg['prop_system']['thrust_target']
    cfg['prop_system'].update(design_mdot_oxidizer=3.185, design_mdot_fuel=1.593)

    def unexpected(*args):
        raise AssertionError('Explicit expansion ratio must bypass inversion')

    monkeypatch.setattr(CEAPropertySource, 'expansion_ratio', unexpected)
    before = deepcopy(cfg)
    with build(cfg) as prop:
        assert prop.throat_area == .0039
        assert prop.expansion_ratio == 4.066
        assert prop.exit_area == pytest.approx(.0039 * 4.066)
        assert prop.design_ambient_pressure == 101325.
        assert (prop.mdot_ox, prop.mdot_fuel) == (3.185, 1.593)
        assert prop.mdot_total == pytest.approx(4.778)
    assert cfg == before
    cfg['prop_system']['thrust_target'] = 1e6
    cfg['engine']['exit_pressure'] = 5e4
    with build(cfg) as prop:
        assert prop.throat_area == .0039 and prop.expansion_ratio == 4.066


@pytest.mark.parametrize('supplied', ['throat_area', 'expansion_ratio'])
def test_partial_engine_inputs_keep_other_sizing_path(supplied):
    cfg = config()
    cfg['engine'][supplied] = .0039 if supplied == 'throat_area' else 4.066
    with build(cfg) as prop:
        assert prop.throat_area == pytest.approx(.0039 if supplied == 'throat_area' else 12000/(2e6*1.5))
        assert prop.expansion_ratio == (4.066 if supplied == 'expansion_ratio' else 5.)


@pytest.mark.parametrize('key,value', [('throat_area', 0.), ('throat_area', float('nan')),
                                     ('expansion_ratio', .5), ('expansion_ratio', float('inf'))])
def test_invalid_engine_inputs_fail(key, value):
    cfg = config()
    cfg['engine'][key] = value
    with pytest.raises(ValueError, match=key):
        build(cfg)


def test_manual_tank_geometry_conserves_fluid_mass_and_matches_the_grid(vespula):
    cfg, vehicle, prop = vespula
    assert vehicle.length == pytest.approx(7.86384, abs=1e-12)
    centers = {'press_tank': 1.8453, 'fuel_tank': 3.7518, 'ox_tank': 5.9083}
    for key, section in vehicle.tanks.items():
        geometry = section.get_fluid_geometry()
        assert geometry.volume == cfg['tanks'][key]['volume']
        assert section.length == pytest.approx(cfg['tanks'][key]['length'])
        assert geometry._profile['height'][-1] == pytest.approx(section.length)
        assert (section.start_station + section.end_station)/2 == pytest.approx(centers[key])
        state = prop.initial_states[key]
        if key == 'press_tank':
            mass = geometry.axial_mass(state['m'])
            expected = state['m']
        else:
            liquid = prop.fluid_properties.state_pt(state['fluid'], state['P'], state['T'])
            mass = geometry.axial_mass(liquid_volume=state['m_liq']/liquid.rho,
                                      liquid_mass=state['m_liq'], ullage_mass=state['m_ull'])
            expected = state['m_liq'] + state['m_ull']
        assert len(mass) == section.n
        assert mass.sum() == pytest.approx(expected)
    # Explicit volume does not depend on sizing density or an ullage factor.
    fuel = vehicle.tanks['fuel_tank']
    fuel.liquid_density *= .8
    assert fuel._tank_volume() == .05976


def test_single_dry_inertia_does_not_change_mass_distribution_or_com(vespula):
    cfg, vehicle, _ = vespula
    assert vehicle.total_mass == pytest.approx(151.)
    assert vehicle.Iyy == pytest.approx(cfg['vehicle']['Iyy'])
    dry_cg, dry_mass = vehicle.cg, vehicle.mass.copy()
    for section in vehicle.sections:
        np.testing.assert_allclose(section.mass, section.mass_inputs['mass_override']/section.n)
    initial_roll = vehicle.Ixx
    supplied = cfg['vehicle'].pop('Iyy')
    vehicle.get_mass_properties()
    assert vehicle.cg == dry_cg
    assert vehicle.Ixx == initial_roll
    np.testing.assert_array_equal(vehicle.mass, dry_mass)
    assert vehicle.Iyy != pytest.approx(supplied)
    cfg['vehicle']['Iyy'] = supplied
    # Move an added fluid mass aft without changing dry cells.
    state = {key: {'tank_id': key, 'axial_mass': np.zeros(section.n)}
             for key, section in vehicle.tanks.items()}
    state['ox_tank']['axial_mass'][-1] = 10.
    vehicle.update_mass_distribution(state)
    fluid_mass = vehicle.mass - dry_mass
    assert vehicle.cg > dry_cg
    assert vehicle.cg == pytest.approx(np.dot(vehicle.mass, vehicle.station)/161.)
    expected = supplied + 151.*(dry_cg - vehicle.cg)**2 + np.dot(fluid_mass, (vehicle.station - vehicle.cg)**2)
    assert vehicle.Iyy == pytest.approx(expected)
    vehicle.get_mass_properties()  # The correction must not accumulate.
    assert vehicle.Iyy == pytest.approx(expected)


def test_aero_uses_existing_geometry_fields_and_optional_exit_only(vespula):
    cfg, vehicle, prop = vespula
    candidate = vehicle.aero_candidate()
    DragModel(cfg['aero']['model']).check(candidate)
    assert candidate['fineness'] == cfg['nosecone']['fineness_ratio']
    assert candidate['tip']*.0254 == pytest.approx(cfg['fin_can']['tip_chord'])
    assert candidate['exit'] == pytest.approx(5.9)
    physical = 2*np.sqrt(prop.exit_area/np.pi)
    assert physical != pytest.approx(cfg['engine']['exit_diameter'])
    del cfg['engine']['exit_diameter']
    assert vehicle.aero_candidate()['exit']*.0254 == pytest.approx(physical)
    assert vehicle.engine.exit_area == prop.exit_area == pytest.approx(.003903218*4.066)
    cfg['engine']['exit_diameter'] = -1
    with pytest.raises(ValueError, match='exit_diameter'):
        vehicle.aero_candidate()


def test_vespula_initializes_with_switch_only_feeds_and_documents_reference_mismatch(vespula):
    cfg, _, prop = vespula
    assert all(b['component'] != 'bang_bang_valve' for b in prop.branch_definitions.values())
    result = prop.network.initialize(bcs={'ambient': {'P': 94000.}})
    assert result['node']['thrust_chamber']['P'] == pytest.approx(cfg['prop_system']['Pc_target'], rel=.1)
    assert result['mdot']['OX_INJ'] == pytest.approx(3.185, rel=.1)
    assert result['mdot']['FUEL_INJ'] == pytest.approx(1.593, rel=.1)
    for key, density in (('ox_tank', 1140.), ('fuel_tank', 804.)):
        state = prop.initial_states[key]
        actual = prop.fluid_properties.state_pt(state['fluid'], state['P'], state['T']).rho
        assert actual == pytest.approx(density, rel=.1)
    # P/T/V remain authoritative: do not force the inconsistent 17.40 kg source mass.
    assert prop.initial_states['press_tank']['m'] == pytest.approx(11.563, rel=1e-3)
    assert abs(prop.initial_states['press_tank']['m']/17.4 - 1) > .1
    for key in ('OX_SWITCH', 'FUEL_SWITCH'):
        assert prop.network.branches[key].sense_node == 'press_tank'
        assert not result['branch'][key]['is_switched']
    first = prop.update(.6, SimpleNamespace(p=94000.), {})
    events = [e for e in first.events if e['component'] == 'OX_RELIEF']
    assert len(events) >= 2
    assert [e['count'] for e in events] == list(range(1, len(events) + 1))
    second = prop.update(.6, SimpleNamespace(p=94000.), {})
    following = [e for e in second.events if e['component'] == 'OX_RELIEF']
    assert following
    assert following[0]['count'] == events[-1]['count'] + 1


@pytest.mark.parametrize('tank,key,value', [('ox_tank','volume',0.), ('fuel_tank','length',-.1),
                                          ('ox_tank','length',.01), ('press_tank','length',.01)])
def test_invalid_manual_geometry_is_rejected(tank, key, value):
    cfg = load_config('Configs/Vespula.yaml')
    cfg['tanks'][tank][key] = value
    pure, _ = property_sources(cfg)
    with pytest.raises(ValueError):
        Vehicle(cfg, pure)
