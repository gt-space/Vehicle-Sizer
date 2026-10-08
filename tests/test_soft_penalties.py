from copy import deepcopy

import pytest

from Fluids.helpers.pressure_tracking import PressureTracking, absolute_linear_integral
from optimizer.core import candidate_score, soft_penalties
from test_optimizer import accepted
from types import SimpleNamespace
from Flight.Flight import FlightSim
from simulation_types import SimResult


def settings():
    return dict(reference_mass=250., constraint_scales={}, unit_scales={'K': 10.},
                soft_penalties=dict(tank_pressure_tracking={
                    'fuel': dict(weight=.05, pressure_scale_pa=1e5, time_scale_s=10.)},
                    max_q=dict(weight=.1, target_pa=140000., scale_pa=10000.)))


def result(mass=250., integral=0., max_q=140000.):
    value = accepted(mass)
    value.max_q = max_q
    value.pressure_tracking = {'fuel': dict(duration_s=40., integral_pa_s=integral)}
    return value


def test_exact_weights_and_mass_tradeoff():
    value = result(integral=1e6, max_q=150000.)
    assert soft_penalties(value, settings()) == {'tank.fuel.pressure_tracking': .05, 'max_q': .1}
    assert candidate_score(value, settings())[0] == pytest.approx(1.15/2.15)
    assert candidate_score(result(275.), settings())[0] < candidate_score(value, settings())[0]
    assert value.accepted


def test_soft_exceedance_does_not_change_feasibility_and_missing_data_errors():
    value = result(integral=1e8, max_q=200000.)
    assert value.accepted and candidate_score(value, settings())[0] < 1
    value.pressure_tracking = {}
    with pytest.raises(ValueError, match='Missing'):
        candidate_score(value, settings())


def test_crossing_and_duration_integrals():
    assert absolute_linear_integral(-10., 10., 2.) == 10.
    assert absolute_linear_integral(10., 10., 2.) == 20.
    assert absolute_linear_integral(-10., -10., 4.) == 40.


def test_tracking_split_invariance_burnout_and_checkpoint():
    definitions = {'node': dict(tank_id='fuel', pressure_tracking_target=100.)}
    whole = PressureTracking(definitions)
    split = PressureTracking(definitions)
    for tracker in (whole, split):
        tracker.accept(0., {'node': {'P': 90.}}, True)
    checkpoint = deepcopy(whole)
    whole.accept(2., {'node': {'P': 110.}}, True)
    split.accept(1., {'node': {'P': 100.}}, True)
    split.accept(2., {'node': {'P': 110.}}, True)
    assert whole.records == split.records
    assert checkpoint.records['fuel']['integral_pa_s'] == 0.
    assert whole.records['fuel']['integral_pa_s'] == 10.
    whole.accept(2., {'node': {'P': 0.}}, False)
    whole.accept(10., {'node': {'P': 1000.}}, False)
    assert whole.records == split.records


@pytest.mark.parametrize('minimum,maximum,penalty', [(.10, .20, 0.), (.09, .20, .03),
    (.10, .21, .03), (.05, .25, .30)])
def test_length_fraction_penalty(minimum, maximum, penalty):
    config = settings()
    config['soft_penalties'] = {'stability_length_fraction': dict(
        lower_target=.1, upper_target=.2, scale=.05, weight=.15)}
    value = result()
    value.min_stability_length_fraction = minimum
    value.max_stability_length_fraction = maximum
    assert soft_penalties(value, config)['stability_length_fraction'] == pytest.approx(penalty)
    assert value.accepted
    assert candidate_score(value, config)[0] == pytest.approx((1+penalty)/(2+penalty))
    value.min_stability_length_fraction = None
    with pytest.raises(ValueError, match='extrema'):
        soft_penalties(value, config)


def test_length_fraction_extrema_use_same_samples_as_calibers():
    flight = SimpleNamespace(cfg={'vehicle': {'OMLD': .5}}, vehicle=SimpleNamespace(cg=2., length=10.))
    result = SimResult()
    for cp in (3., 4., 2.5):
        FlightSim._record_extrema(flight, result, SimpleNamespace(q=1.), SimpleNamespace(cp=cp),
                               SimpleNamespace(alpha=0., t=0.))
    assert result.min_stability_length_fraction == .05
    assert result.max_stability_length_fraction == .2
    assert result.min_stability_calibers == 1.


@pytest.mark.parametrize('field,value', [('scale', 0.), ('weight', -.1),
    ('upper_target', .05), ('lower_target', float('nan'))])
def test_invalid_length_fraction_policy(field, value):
    spec = dict(lower_target=.1, upper_target=.2, scale=.05, weight=.15)
    spec[field] = value
    with pytest.raises(ValueError, match='Invalid stability'):
        soft_penalties(result(), {'soft_penalties': {'stability_length_fraction': spec}})
