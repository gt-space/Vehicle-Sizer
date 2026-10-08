"""GG design calculations and fixed-demand conservation/dryout regressions."""
from copy import deepcopy

import pytest

from Fluids.design import size_gas_generator
from Fluids.helpers.templates import load_template
from test_electric_pump import config, build
from test_fluid_network import wet_tank, Properties
from Fluids import FluidNetwork


def definition(**overrides):
    return dict(efficiency=.6, turbine_outlet_pressure_pa=1e5,
                mixture_ratio=.5, stiffness=.2, **overrides)


def test_turbine_power_and_placeholder_cp():
    result = size_gas_generator(10000., 1e6, definition(), dict(T=1000., gamma=1.4, R=300.))
    assert result['cp_j_kg_k'] == pytest.approx(1050.)
    delta = 1000 * (1 - .1 ** (2 / 7))
    assert result['design_mdot'] == pytest.approx(10000 / (.6 * 1050 * delta))
    assert result['fuel_mdot'] + result['oxidizer_mdot'] == pytest.approx(result['design_mdot'])
    assert result['oxidizer_mdot'] / result['fuel_mdot'] == pytest.approx(.5)


@pytest.mark.parametrize('key,value', [('efficiency', 0), ('efficiency', 1.1),
    ('turbine_outlet_pressure_pa', 1e6), ('mixture_ratio', -1)])
def test_invalid_turbine_inputs(key, value):
    d = definition()
    d[key] = value
    with pytest.raises(ValueError):
        size_gas_generator(10000., 1e6, d, dict(T=1000., gamma=1.4, R=300.))


def gg_config():
    cfg = config()
    template = load_template(cfg)
    cfg['prop_system']['template'] = template
    for pump in cfg['prop_system']['pumps'].values():
        pump['drive'] = 'gas_generator'
        del pump['max_power_kw']
    for role, tank in [('fuel', 'fuel_ullage'), ('oxidizer', 'ox_ullage')]:
        template['branches'][f'GG_{role}'] = dict(component='mass_flow', circuit=role,
            **{'from': tank, 'to': 'ambient'}, from_port='liquid')
    cfg['prop_system']['gas_generators'] = {'main': dict(definition(),
        pumps=['fuel_pump', 'oxidizer_pump'], fuel_branch='GG_fuel', oxidizer_branch='GG_oxidizer')}
    return cfg


def test_gg_sizing_is_order_independent_and_does_not_mutate_config():
    cfg = gg_config()
    original = deepcopy(cfg)
    first = build(cfg)
    assert cfg == original
    cfg['prop_system']['template']['branches'] = dict(reversed(list(cfg['prop_system']['template']['branches'].items())))
    second = build(cfg)
    assert first.gg_sizing == second.gg_sizing
    sizing = first.gg_sizing['main']
    assert sizing['required_shaft_power_w'] == pytest.approx(sum(p['required_shaft_power_w'] for p in first.pump_sizing.values()))
    for role in ('fuel', 'oxidizer'):
        branch = first.branch_definitions[f'GG_{role}']
        assert branch['target_mdot'] == sizing[f'{role}_mdot']
        assert branch['design_mdot'] == branch['target_mdot']
        assert branch['required_phase'] == 'liquid'
    assert first.battery_sizing == {}


def test_missing_drive_assignment_rejected():
    cfg = gg_config()
    cfg['prop_system']['gas_generators'] = {}
    with pytest.raises(ValueError, match='Every gas_generator pump'):
        build(cfg)


def test_independent_fixed_drains_stop_on_dryout_and_preserve_other_rate():
    nodes = {'a': wet_tank(mass=.02), 'b': wet_tank(mass=.2),
             'sink': dict(component='boundary', P=1e5)}
    branches = {f'drain_{name}': dict(component='mass_flow', **{'from': name, 'to': 'sink'},
                          from_port='liquid', target_mdot=.1, design_mdot=.1,
                          required_phase='liquid') for name in ('a', 'b')}
    network = FluidNetwork(nodes, branches, fluid_properties=Properties(), stop_at_shutdown=False)
    try:
        network.initialize()
        initial_b = network.state.nodes['b']['m_liq']
        network.update(.1)
        assert network.state.nodes['a']['m_liq'] == pytest.approx(.01)
        assert network.state.nodes['b']['m_liq'] == pytest.approx(initial_b - .01)
        # Incompressible fixture: liquid energy tracks the remaining mass.
        assert network.state.nodes['b']['U_liq'] == pytest.approx((initial_b - .01) * 600000.)
        preview = network.update(.2, commit=False)
        assert preview["mdot"]["drain_a"] == pytest.approx(0., abs=1e-8)
        assert network.branches["drain_a"].active
        network.update(.2)
        assert not network.branches['drain_a'].active
        assert network.branches['drain_a'].enabled
        assert network.state.branches['drain_a'].mdot == pytest.approx(0., abs=1e-8)
        assert network.state.branches['drain_b'].mdot == pytest.approx(.1)
        assert network.state.nodes['b']['m_liq'] == pytest.approx(initial_b - .03)
    finally:
        network.close()


def test_gg_sizing_uses_existing_table_source_and_rejects_out_of_bounds_mr():
    from pathlib import Path
    from FluidTables.PropertyModels import TableCombustionPropertySource
    from Fluids.PropSystem import PropSystem
    import propulsion_fixtures as support
    source = TableCombustionPropertySource(Path(__file__).resolve().parents[1] / 'FluidTables/sizer_lookups.h5', 1)
    cfg = gg_config()
    ox, fuel, _ = support.tanks()
    tanks = {'ox_tank': ox, 'fuel_tank': fuel}
    with PropSystem(cfg, tanks, combustion_properties=source) as system:
        sizing = system.gg_sizing['main']
        properties = source.evaluate(sizing['inlet_pressure_pa'], .5,
                                     system.design_ambient_pressure, system.expansion_ratio)
        assert sizing['inlet_temperature_k'] == pytest.approx(properties.T)
        assert sizing['gamma'] == pytest.approx(properties.gamma)
        assert sizing['R'] == pytest.approx(properties.R)
    cfg['prop_system']['gas_generators']['main']['mixture_ratio'] = .01
    with pytest.raises(ValueError):
        PropSystem(cfg, tanks, combustion_properties=source)


def test_pressurant_sizing_includes_gg_and_is_order_independent():
    from unittest.mock import patch
    import propulsion_fixtures as support
    from test_prop_system import config as regulated_config
    from Fluids.PropSystem import PropSystem
    cfg = regulated_config('pump_fed', 'regulator')
    original_template = load_template(cfg)
    with patch('Fluids.PropSystem._make_cea', return_value=support.FakeCEA()):
        tanks = dict(zip(('ox_tank', 'fuel_tank', 'press_tank'), support.tanks()))
        baseline = PropSystem(cfg, tanks)
        cfg['prop_system']['template'] = original_template
        gg = gg_config()['prop_system']
        cfg['prop_system']['pumps'] = gg['pumps']
        cfg['prop_system']['gas_generators'] = gg['gas_generators']
        for role, tank in [('fuel', 'fuel_ullage'), ('oxidizer', 'ox_ullage')]:
            original_template['branches'][f'GG_{role}'] = dict(component='mass_flow', circuit=role,
                **{'from': tank, 'to': 'ambient'}, from_port='liquid')
        cfg['prop_system']['template']['branches'] = dict(reversed(list(original_template['branches'].items())))
        sized = PropSystem(cfg, tanks)
        for role, regulator, engine_mdot in [('fuel', 'FUEL_REGULATOR', baseline.mdot_fuel),
                                           ('oxidizer', 'OX_REGULATOR', baseline.mdot_ox)]:
            ratio = 1 + sized.gg_sizing['main'][f'{role}_mdot'] / engine_mdot
            assert sized.branch_definitions[regulator]['CdA'] == pytest.approx(baseline.branch_definitions[regulator]['CdA'] * ratio)


def test_other_drain_continues_after_main_engine_shutdown():
    from test_fluid_network import propulsion, Combustion
    base = propulsion(stop_at_shutdown=False)
    nodes, branches = deepcopy(base.node_definitions), deepcopy(base.branch_definitions)
    base.close()
    for name in ('ox', 'fuel'):
        branches[f'drain_{name}'] = dict(component='mass_flow', **{'from': name, 'to': 'ambient'},
            from_port='liquid', required_phase='liquid', target_mdot=.01, design_mdot=.01)
    with FluidNetwork(nodes, branches, fluid_properties=Properties(), combustion_properties=Combustion(),
                      stop_at_shutdown=False, tolerances={'max_step': .02}) as net:
        net.initialize()
        for _ in range(100):
            result = net.update(.02)
            if result['node']['chamber']['mode'] == 'shutdown':
                break
        assert result['node']['chamber']['mode'] == 'shutdown'
        assert result['mdot']['drain_ox'] == pytest.approx(0., abs=1e-8)
        assert result['mdot']['drain_fuel'] == pytest.approx(.01)
        assert net.update(.02)['mdot']['drain_fuel'] == pytest.approx(.01)
