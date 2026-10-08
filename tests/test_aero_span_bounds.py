"""Diameter-dependent fin geometry through the actual expanded aero deck."""
from copy import deepcopy

import numpy as np
import pytest

import optimizer.core as opt
from Configs.loader import load_config
from Flight.flight_forces import Aero
from Vehicle.Engine import Engine
from Vehicle.Vehicle import Vehicle
from Vehicle.sections.FinCan import FinCan
from test_aero import CANDIDATE


@pytest.fixture(scope='module')
def model():
    return opt.DragModel()


@pytest.mark.parametrize('diameter', [8., 12., 16.])
@pytest.mark.parametrize('ratio', [.5, 1.2])
def test_expanded_span_geometry_export_and_aero_closure(model, diameter, ratio):
    cfg = load_config('Configs/flight_pump_fed_regulator.yaml')
    cfg['vehicle']['OMLD'] = diameter * opt.INCH
    cfg['fin_can'].update(span=ratio*diameter*opt.INCH,
                         boattail_aft_diameter=7.5*opt.INCH,
                         boattail_length=22.*opt.INCH,
                         root_chord=18.*opt.INCH, tip_chord=6.*opt.INCH)
    engine = Engine(30., .5, np.pi*(6.*opt.INCH)**2/4)
    fin = FinCan(cfg, engine)
    vehicle = Vehicle.__new__(Vehicle)
    vehicle.cfg, vehicle.engine, vehicle.length, vehicle.sections = cfg, engine, 360.*opt.INCH, [fin]
    candidate = vehicle.aero_candidate()
    assert candidate['span'] == pytest.approx(ratio*diameter)
    assert candidate['root'] == pytest.approx(18.)  # Chords are still physical lengths.
    assert fin.fin_area == pytest.approx(.5*(18.+6.)*ratio*diameter*opt.INCH**2)
    assert fin._get_fin_mass() > 0
    model.check(candidate)
    assert opt.domain_records(candidate, model)['aero.span'].margin == pytest.approx(0., abs=1e-12)
    aero = Aero({'fins_on_boattail': True}, candidate, model)
    for mach in [.1, .97, 2., 10.]:
        for alpha in [0., 5., 15.]:
            coefficients = aero.coefficients(mach, np.radians(alpha), True)
            assert np.isfinite(coefficients).all()
    axial = aero.axial_distribution(2., np.radians(5.), True)
    normal = aero.normal_distribution(2., np.radians(5.))
    cd, ca, cn, cp = aero.coefficients(2., np.radians(5.), True)
    assert np.trapezoid(axial['dca_dx'], axial['x']) == pytest.approx(ca, abs=1e-10)
    assert np.trapezoid(normal['dcn_dx'], normal['x']) == pytest.approx(cn, abs=1e-10)


@pytest.mark.parametrize('diameter', [8., 12., 16.])
@pytest.mark.parametrize('ratio', [.49, 1.21])
def test_outside_diameter_relative_span_has_signed_metric_margin(model, diameter, ratio):
    candidate = {**CANDIDATE, 'omld': diameter, 'boattail_aft': 7.5,
                 'exit': 6., 'span': diameter*ratio}
    record = opt.domain_records(candidate, model)['aero.span']
    assert record.margin == pytest.approx(-.01*diameter*opt.INCH)
    assert record.units == 'm'
    with pytest.raises(ValueError, match='span'):
        model.check(candidate)


@pytest.mark.parametrize('settings_path', ['Configs/optimizer.yaml', 'Configs/optimizer_pressure_fed.yaml'])
def test_conditional_span_tracks_diameter_and_respects_user_bounds(model, settings_path, tmp_path):
    cfg, settings = opt.prepare(load_config(settings_path), model)
    assert opt.aero_bounds(model)['span'] == pytest.approx((4.*opt.INCH, 19.2*opt.INCH))
    pure, combustion = opt.property_sources(cfg)
    evaluator = opt.Evaluator(cfg, settings, model, pure, combustion, tmp_path)
    bounds = np.array(evaluator.bounds)
    for fraction in [0., .5, 1.]:
        values = bounds.mean(axis=1)
        i = evaluator.names.index('omld')
        values[i] = bounds[i, 0] + fraction*np.ptp(bounds[i])
        for span_fraction in [0., .5, 1.]:
            j = evaluator.names.index('span')
            values[j] = bounds[j, 0] + span_fraction*np.ptp(bounds[j])
            candidate = evaluator.decode(values)
            diameter = candidate['vehicle']['OMLD']
            expected_ratio = .5 + .7*span_fraction
            assert candidate['fin_can']['span'] == pytest.approx(expected_ratio*diameter)
            assert bounds[j, 0] <= candidate['fin_can']['span'] <= bounds[j, 1]
    # A user's narrower physical interval remains a hard limit.
    narrowed = deepcopy(settings)
    narrowed['bounds']['span'] = [.21, .22]
    limited = opt.Evaluator(cfg, narrowed, model, pure, combustion, tmp_path)
    candidate = limited.decode(np.mean(limited.bounds, axis=1))
    assert .21 <= candidate['fin_can']['span'] <= .22
    # Valid global limits can still have no overlap at the selected diameter.
    narrowed['bounds']['span'] = [.1016, .11]
    limited = opt.Evaluator(cfg, narrowed, model, pure, combustion, tmp_path)
    with pytest.raises(opt.GeometryError) as caught:
        limited.decode(np.mean(limited.bounds, axis=1))
    assert caught.value.constraints['geometry.bounds.span'] < 0


def test_si_round_trip_does_not_reject_exact_span_boundaries(model):
    for diameter_m in np.linspace(.2032, .4064, 101):
        diameter = diameter_m / opt.INCH
        for ratio in [.5, 1.2]:
            candidate = {**CANDIDATE, 'omld': diameter, 'boattail_aft': 7.5, 'exit': 6.,
                         'span': (ratio*diameter*opt.INCH)/opt.INCH}
            model.check(candidate)
            assert opt.domain_records(candidate, model)['aero.span'].margin >= 0


@pytest.mark.parametrize('architecture', ['pump_fed', 'pressure_fed'])
@pytest.mark.parametrize('controller', ['regulator', 'bang_bang'])
def test_current_configs_size_and_export_valid_new_deck_geometry(model, architecture, controller):
    cfg = load_config(f'Configs/flight_{architecture}_{controller}.yaml')
    pure, combustion = opt.property_sources(cfg)
    vehicle = Vehicle(cfg, pure)
    with opt.PropSystem(cfg, vehicle.tanks, pure, combustion) as prop:
        vehicle.build(prop)
        candidate = vehicle.aero_candidate()
        model.check(candidate)
        assert all(record.margin >= 0 for record in opt.domain_records(candidate, model).values())
        aero = Aero(cfg['aero'], candidate, model)
        assert np.isfinite(aero.coefficients(2., np.radians(5.), True)).all()
