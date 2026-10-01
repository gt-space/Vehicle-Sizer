from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from Fluids.helpers.pump_curve import scaled_pump_curve
from Fluids.FluidState import FluidState
from test_electric_pump import build, config

POINTS = [[0, 3600000], [1.09, 3450000], [1.6, 3160000],
          [2.172, 2548600], [2.8, 1830000], [3.25, 1250000], [4.2, 0]]
REFERENCE = dict(reference_mdot=2.172, points=POINTS)


def test_corrected_workbook_fit_and_exact_design_scaling():
    curve = scaled_pump_curve(3, 5e6, **REFERENCE)
    coefficients = np.polyfit(np.array(POINTS)[:, 0], np.array(POINTS)[:, 1], 2)
    assert coefficients == pytest.approx([-192693.0860659682, -87152.6713148693, 3676010.071294608])
    equivalent = scaled_pump_curve(3, 5e6, reference_mdot=2.172, coefficients=coefficients)
    for flow in (0, 1, 3, 4):
        assert curve(flow) == pytest.approx(equivalent(flow))
    assert curve(3) == pytest.approx(5e6)
    assert curve(0) > curve(3) > curve(4) > 0
    assert curve(curve.max_mdot) == pytest.approx(0, abs=1e-8)
    for flow in (-1, curve.max_mdot * 1.01, float('nan')):
        with pytest.raises(ValueError):
            curve.validate_flow(flow)
    assert scaled_pump_curve(6, 10e6, **REFERENCE)(8) == pytest.approx(2 * curve(4))


@pytest.mark.parametrize('kwargs', [dict(points=[[0, 1], [1, 1]]),
    dict(coefficients=[1, 0, 1]), dict(coefficients=[-1, float('nan'), 5]),
    dict(points=POINTS, coefficients=[-1, 0, 5]), dict(points=[[0,1],[0,2],[2,3]])])
def test_bad_reference_rejected(kwargs):
    with pytest.raises(ValueError):
        scaled_pump_curve(3, 5e6, reference_mdot=2.172, **kwargs)


def test_network_receives_fresh_curve_and_evaluates_off_design():
    cfg = config()
    for pump in cfg['prop_system']['pumps'].values():
        pump['curve'] = deepcopy(REFERENCE)
    original = deepcopy(cfg)
    system = build(cfg)
    assert cfg == original
    branch = system.network.branches['OX_PUMP']
    curve = branch.parameters['head_model']
    design = system.pump_sizing['oxidizer_pump']
    mdot = design['design_mdot']
    assert curve(mdot) == pytest.approx(design['pressure_rise_pa'])
    flow = 1.4 * mdot
    source = {'ox': FluidState('Oxygen', 'liquid', {'rho': 1000., 'h': 1e5})}
    def residual(head):
        nodes = {branch.from_node: {'P': 2e6}, branch.to_node: {'P': 2e6 + head}}
        trial = branch.evaluate({'mdot': flow}, nodes, source, {'main': 1})
        return branch.residual(trial, nodes)[0]
    assert residual(curve(flow)) == pytest.approx(0.)
    assert abs(residual(design['pressure_rise_pa'])) > .1
    output = system.update(None, SimpleNamespace(p=1e5), {})
    runtime = output.branch['OX_PUMP']
    assert runtime['dP'] == pytest.approx(-curve(output.mdot['OX_PUMP']), rel=1e-6)
    cfg['prop_system']['pumps']['oxidizer_pump']['pressure_rise_pa'] = 1.2e6
    changed = build(cfg).network.branches['OX_PUMP'].parameters['head_model']
    assert changed is not curve
    assert changed(mdot) == pytest.approx(1.2e6)


def test_accepted_liquid_curve_domain_checked_but_gas_bypass_unchanged():
    cfg = config()
    cfg['prop_system']['pumps']['oxidizer_pump']['curve'] = deepcopy(REFERENCE)
    system = build(cfg)
    system.update(None, SimpleNamespace(p=1e5), {})
    state = deepcopy(system.network.state)
    branch = state.branches['OX_PUMP']
    branch.trial_values['mdot'] = 1e6
    for flow in branch.flows.values():
        flow['mdot'] = 1e6
    with pytest.raises(ValueError, match='outside'):
        system._constraint_margins(state)
    branch.properties['pump_active'] = False
    system._constraint_margins(state)
