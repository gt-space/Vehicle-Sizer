"""Pressure-drop and CdA inputs produce consistent design and runtime sizing."""
from copy import deepcopy
import pytest
from Configs.loader import load_config
from Fluids.design import prepare_design_inputs
from diagnostics.errors import LookupBoundsError
from Fluids.helpers import templates
from Fluids.PropSystem import PropSystem
from FluidTables.PropertyModels import property_sources
from Vehicle.Vehicle import Vehicle



def _sized_feed_config(cfg, pure, combustion):
    template, states, walls = prepare_design_inputs(
        cfg, fluid_properties=pure, combustion_properties=combustion)
    vehicle = Vehicle(cfg, pure, initial_states=states, wall_pressures=walls)
    with PropSystem(cfg, vehicle.tanks, pure, combustion,
                    template=template, initial_states=states) as propulsion:
        vehicle.build(propulsion)
        return template, states, walls, vehicle.total_mass, {
            key: branch['CdA'] for key, branch in propulsion.branch_definitions.items()
            if key in ('OX_TANK_INJ', 'FUEL_TANK_INJ')}


@pytest.mark.parametrize('legs', [('ox',), ('fuel',), ('ox', 'fuel')])
@pytest.mark.parametrize('override_pressure', [False, True])
def test_cda_sizing_round_trip_matches_dp_and_preserves_inputs(legs, override_pressure):
    cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
    if override_pressure:
        cfg['prop_system']['initial_conditions']['ox_tank']['P'] = 3e6
        cfg['prop_system']['initial_conditions']['fuel_tank']['P'] = 3e6
    pure, combustion = property_sources(cfg)
    original = _sized_feed_config(cfg, pure, combustion)
    for leg in legs:
        cfg['prop_system'].pop(f'{leg}_tank_inj_dp')
        cfg['prop_system'][f'{leg}_system_cda'] = original[4][leg.upper() + '_TANK_INJ']
    before = deepcopy(cfg)
    actual = _sized_feed_config(cfg, pure, combustion)
    assert cfg == before
    assert actual[2] == pytest.approx(original[2], rel=1e-9)
    assert actual[3] == pytest.approx(original[3], rel=1e-9)
    assert actual[4] == pytest.approx(original[4], rel=1e-9)
    for tank in ('ox_tank', 'fuel_tank'):
        assert actual[1][tank]['P'] == pytest.approx(original[1][tank]['P'], rel=1e-9)


@pytest.mark.parametrize('dp,cda', [(1e5, 1e-4), (None, None), (-1, None),
                                  (None, 0), (None, float('nan'))])
def test_feed_sizing_rejects_invalid_or_conflicting_inputs(dp, cda):
    cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
    cfg['prop_system'].update(ox_tank_inj_dp=dp, ox_system_cda=cda)
    with pytest.raises(ValueError, match='exactly one|finite and positive'):
        prepare_design_inputs(cfg)


def test_smaller_cda_increases_pressure_and_checks_template_conflicts():
    cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
    pure, combustion = property_sources(cfg)
    baseline = _sized_feed_config(cfg, pure, combustion)
    cfg['prop_system']['ox_tank_inj_dp'] = None
    cfg['prop_system']['ox_system_cda'] = baseline[4]['OX_TANK_INJ'] / 2
    actual = _sized_feed_config(cfg, pure, combustion)
    assert actual[2]['ox_tank'] > baseline[2]['ox_tank']
    assert actual[3] > baseline[3]
    cfg['prop_system']['template'] = templates.read_template(cfg)
    cfg['prop_system']['template']['branches']['OX_TANK_INJ']['CdA'] = 1e-4
    with pytest.raises(ValueError, match='not also'):
        prepare_design_inputs(cfg, fluid_properties=pure, combustion_properties=combustion)


def test_cda_sizing_rejects_pressure_outside_property_domain():
    cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
    cfg['prop_system'].pop('ox_tank_inj_dp')
    cfg['prop_system']['ox_system_cda'] = 1e-12
    pure, combustion = property_sources(cfg)
    with pytest.raises(LookupBoundsError, match='pressure domain'):
        prepare_design_inputs(cfg, fluid_properties=pure, combustion_properties=combustion)


def test_optimizer_decodes_cda_config_using_the_same_pressure_sizing(tmp_path):
    import numpy as np
    from optimizer import core as opt
    model = opt.DragModel()
    cfg, settings = opt.prepare(load_config('Configs/optimizer_pressure_fed.yaml'), model)
    pure, combustion = property_sources(cfg)
    for leg, cda in [('ox', .00032706944491738433), ('fuel', .0001776871392414737)]:
        cfg['prop_system'].pop(f'{leg}_tank_inj_dp')
        cfg['prop_system'][f'{leg}_system_cda'] = cda
    evaluator = opt.Evaluator(cfg, settings, model, pure, combustion, tmp_path)
    candidate = evaluator.decode(np.mean(evaluator.bounds, axis=1))
    template, states, walls, mass, cdas = _sized_feed_config(candidate, pure, combustion)
    assert mass > 0
    assert cdas['OX_TANK_INJ'] == cfg['prop_system']['ox_system_cda']
    assert walls['ox_tank'] == pytest.approx(1.1 * states['ox_tank']['P'])
