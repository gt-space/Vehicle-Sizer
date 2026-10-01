"""Connected-network IDAS regression tests, including real discrete transitions."""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip('sundials4py')

from Fluids import FluidNetwork, TrialDomainError
from FluidTables.PropertyModels import PureFluidProperties, CombustionProperties


@dataclass
class Geometry:
    volume: float = 0.01

    def fill_state(self, liquid_volume):
        return dict(fill_height=liquid_volume, liquid_contact_area=liquid_volume,
                    ullage_contact_area=self.volume - liquid_volume)

    def axial_mass(self, mass=None, *, liquid_volume=None, liquid_mass=None, ullage_mass=None):
        return [mass] if mass is not None else [liquid_mass, ullage_mass]


class Properties:
    def state_pt(self, fluid, pressure, temperature):
        gas = fluid == 'gas'
        rho = pressure / (300 * temperature) if gas else 1000.
        u = (750. if gas else 2000.) * temperature
        return PureFluidProperties(pressure, temperature, rho, u + pressure / rho, u,
                                   300., 1.4, 1e-5, 0.03, 1050., 1 / temperature)

    def derivatives_pt(self, fluid, pressure, temperature):
        if fluid != 'gas':
            return dict(drho_dP=0., drho_dT=0., du_dP=0., du_dT=2000.)
        return dict(drho_dP=1 / (300 * temperature), drho_dT=-pressure / (300 * temperature**2),
                    du_dP=0., du_dT=750.)

    def state_bounds(self, fluid):
        return (1., 1e9), (1., 1e5)

    def supports_saturation(self, fluid):
        return False


class Combustion:
    def evaluate(self, **kwargs):
        return CombustionProperties(1500., 1.5, 350., 1.2, 3000.)


def gas_tank(p=9e5, volume=0.1):
    m = p * volume / 90000.
    return dict(component='volume', fluid='gas', geometry=Geometry(volume),
                state0=dict(P=p, T=300., m=m, U=m * 225000.))


def wet_tank(fluid='ox', mass=0.2, pressure=1e6):
    mg = (0.01 - mass / 1000.) * pressure / 90000.
    return dict(component='propellant_tank', liquid_fluid=fluid, gas_fluid='gas', geometry=Geometry(),
                state0=dict(P=pressure, T=300., gas_T=300., m_liq=mass, U_liq=mass * 600000.,
                            m_ull=mg, U_ull=mg * 225000.))


def loss(source, target, cda=1e-5, **kwargs):
    return dict(component='loss', **{'from': source, 'to': target}, CdA=cda, **kwargs)


def blowdown(**options):
    return FluidNetwork({'tank': gas_tank(), 'ambient': dict(component='boundary', P=1e5)},
                        {'outlet': loss('tank', 'ambient')}, fluid_properties=Properties(), **options)


def propulsion(*, simultaneous=False, pump=False, **options):
    nodes = {'ox': wet_tank('ox', 0.2), 'fuel': wet_tank('fuel', 0.2 if simultaneous else 0.6),
             'ambient': dict(component='boundary', P=1e5),
             'chamber': dict(component='combustor', P0=3e5, oxidizer_fluid='ox', fuel_fluid='fuel',
                             combustion_fluid='products', ambient_node='ambient', expansion_ratio=5.,
                             cstar_efficiency=1., cf_efficiency=1.)}
    branches = {'ox_feed': loss('ox', 'chamber', from_port='liquid'),
                'fuel_feed': loss('fuel', 'chamber', from_port='liquid'),
                'nozzle': dict(component='nozzle', **{'from': 'chamber', 'to': 'ambient'}, At=0.002, Cd=1.)}
    if pump:
        for name in ('ox_feed', 'fuel_feed'):
            branches[name].update(component='pump', dP=2e5, gas_CdA=1e-5)
    return FluidNetwork(nodes, branches, fluid_properties=Properties(), combustion_properties=Combustion(), **options)


def test_analytic_blowdown_and_named_equations():
    with blowdown() as net:
        first = net.initialize()
        q0 = first['mdot']['outlet']
        result = net.update(20.)
        ratio = (1 + 0.2 * q0 * 20.) ** -5
        assert result['node']['tank']['m'] == pytest.approx(ratio, rel=2e-6)
        assert result['node']['tank']['P'] == pytest.approx(9e5 * ratio**1.4, rel=2e-6)
        assert len(net.variable_index) == len(net.equation_index) == 5
        assert net.differential.tolist() == [1, 1, 0, 0, 0]
        assert net.session.statistics()['jacobians'] > 0


@pytest.mark.parametrize('initially_open', [True, False])
def test_multiple_bang_bang_events_and_preview(initially_open):
    nodes = {'supply': gas_tank(2e6), 'tank': gas_tank(3e5, .01),
             'ambient': dict(component='boundary', P=1e5)}
    branches = {'valve': dict(component='bang_bang_valve', **{'from': 'supply', 'to': 'tank'},
                             CdA=3e-5, target_pressure=3e5, pressure_band=2e4, initially_open=initially_open),
                'outlet': loss('tank', 'ambient', 1e-5)}
    with FluidNetwork(nodes, branches, fluid_properties=Properties()) as net:
        net.initialize()
        y, stats = net.y.copy(), net.session.statistics()
        prediction = net.update(4., commit=False)
        np.testing.assert_array_equal(net.y, y)
        assert net.time == 0 and net.events == [] and net.session.statistics() == stats
        result = net.update(4.)
        events = [e for e in result['events'] if e['name'] == 'switch']
        assert len(events) > 5
        assert all(a['is_open'] != b['is_open'] for a, b in zip(events, events[1:]))
        assert [e['time_s'] for e in events] == sorted(e['time_s'] for e in events)
        assert result['node']['tank']['P'] == pytest.approx(prediction['node']['tank']['P'], rel=2e-5)


@pytest.mark.parametrize('pump', [False, True])
@pytest.mark.parametrize('simultaneous', [False, True])
def test_dryout_shutdown_and_continued_blowdown(pump, simultaneous):
    with propulsion(pump=pump, simultaneous=simultaneous, tolerances={'max_step': .05}) as net:
        initial = net.initialize()
        assert initial['node']['chamber']['mode'] == 'combusting'
        result = net.update(3.)
        dry = [e for e in result['events'] if e['name'] == 'dryout']
        assert len(dry) == 2
        assert result['node']['chamber']['mode'] == 'shutdown'
        assert result['mdot']['nozzle'] > 0
        assert all(net.branches[k].phase == 'gas' for k in ('ox_feed', 'fuel_feed'))
        assert len(result['numerical_remainders']) == 2
        assert sum(r['mass'] for r in result['numerical_remainders']) <= 2e-5
        assert len(net.variable_index) == len(net.equation_index)
        prior = result['node']['ox']['mass']
        assert net.update(.2)['node']['ox']['mass'] < prior
        if simultaneous:
            assert dry[0]['time_s'] == pytest.approx(dry[1]['time_s'], abs=1e-6)
        else:
            assert dry[0]['time_s'] < dry[1]['time_s']


def test_regulator_holds_target_and_hits_capacity():
    nodes = {'supply': gas_tank(1e6, .005), 'tank': gas_tank(3e5, .01),
             'ambient': dict(component='boundary', P=1e5)}
    branches = {'regulator': dict(component='regulator', **{'from': 'supply', 'to': 'tank'},
                                 CdA=1e-5, target_pressure=3e5),
                'outlet': loss('tank', 'ambient', 1e-5)}
    # This long capacity/reversal test checks strict closure at event roots.
    with FluidNetwork(nodes, branches, fluid_properties=Properties(),
                      tolerances={'rtol': 1e-7}) as net:
        result = net.initialize()
        assert net.regulator_modes['regulator'] == 'regulating'
        assert result['mdot']['regulator'] == pytest.approx(result['mdot']['outlet'], rel=1e-6)
        result = net.update(.2)
        assert result['node']['tank']['P'] == pytest.approx(3e5, abs=.1)
        result = net.update(10.)
        assert any(e['name'] == 'capacity_limit' for e in result['events'])
        assert result['node']['tank']['P'] < 3e5


def test_failed_preview_and_commit_preserve_accepted_state():
    with blowdown() as net:
        net.update(1.)
        y, dy, time = net.y.copy(), net.ydot.copy(), net.time
        def broken(t):
            raise ValueError('intentional boundary failure')
        with pytest.raises(ValueError, match='intentional boundary failure'):
            net.update(1., bcs={'ambient': broken}, commit=False)
        np.testing.assert_array_equal(net.y, y)
        assert net.session is not None
        session = net.session
        with pytest.raises(ValueError, match='intentional boundary failure'):
            net.update(1., bcs={'ambient': broken})
        np.testing.assert_array_equal(net.y, y)
        np.testing.assert_array_equal(net.ydot, dy)
        assert net.time == time and net.session is None
        assert session.owner is session.context is session.y is session.matrix is None
        import gc
        gc.collect()  # Closed native resources must not depend on GC order.
        assert net.update(1.)['time'] == time + 1


@pytest.mark.parametrize('rtol', [1e-7, 1e-4])
def test_capacity_mode_does_not_cycle_on_sub_tolerance_pressure_drift(rtol):
    target = 3e5
    nodes = {'supply': gas_tank(4e5), 'tank': gas_tank(target*(1+.5*rtol), .01),
             'ambient': dict(component='boundary', P=1e5)}
    branches = {'regulator': dict(component='regulator', **{'from': 'supply', 'to': 'tank'},
                                  CdA=1e-8, target_pressure=target),
                'outlet': loss('tank', 'ambient', 1e-5)}
    with FluidNetwork(nodes, branches, fluid_properties=Properties(), tolerances={'rtol': rtol}) as net:
        net.initialize()
        # Reproduce an exhausted regulator with a tiny positive pressure offset
        # left by the preceding ideal-regulation interval.
        net.regulator_modes['regulator'] = 'capacity'
        net._new_session()
        before = len(net.events)
        net._settle_initial_events()
        assert net.regulator_modes['regulator'] == 'capacity'
        assert len(net.events) == before
        assert net.ydot[net.variable_index['tank.m']] < 0  # Actual supply deficit.


def test_recoverable_trial_error_retries_without_changing_modes():
    with blowdown() as net:
        net.initialize()
        original = net.session.residual
        injected = []
        def residual(t, y, dy, out):
            if t > .5 and not injected:
                injected.append(t)
                raise TrialDomainError('one-shot numerical trial error')
            original(t, y, dy, out)
        net.session.residual = residual
        net.update(2.)
        assert injected and net.session.recoverable_errors == 1
        assert not net.events


def test_closed_transfer_conserves_mass_and_energy():
    with FluidNetwork({'a': gas_tank(9e5), 'b': gas_tank(3e5)},
                      {'line': loss('a', 'b')}, fluid_properties=Properties()) as net:
        before = net.initialize()
        after = net.update(5.)
        for quantity in ('m', 'U'):
            total = lambda result: sum(result['node'][key][quantity] for key in ('a', 'b'))
            assert total(after) == pytest.approx(total(before), rel=1e-10)


def test_property_limit_is_named_and_rolls_back():
    class Limited(Properties):
        def state_bounds(self, fluid):
            return (8e5, 1e9), (1., 1e5)
        def state_pt(self, fluid, pressure, temperature):
            if pressure < 8e5:
                raise AssertionError('A property call escaped its declared domain')
            return super().state_pt(fluid, pressure, temperature)
    with FluidNetwork({'tank': gas_tank(), 'ambient': dict(component='boundary', P=1e5)},
                      {'outlet': loss('tank', 'ambient')}, fluid_properties=Limited()) as net:
        net.initialize()
        y = net.y.copy()
        with pytest.raises(RuntimeError, match='tank.pressure_low'):
            net.update(20.)
        np.testing.assert_array_equal(net.y, y)
        assert net.time == 0 and net.events == []


def test_transport_cycle_is_explicit():
    nodes = {k: dict(component='junction', fluid='gas', phase='gas', state0=dict(P=2e5,T=300.)) for k in ('a', 'b')}
    with FluidNetwork(nodes, {'ab': loss('a', 'b'), 'ba': loss('b', 'a')}, fluid_properties=Properties()) as net:
        with pytest.raises(ValueError, match='transport cycle'):
            net.initialize()
        assert net.y is None


def test_continuous_flow_reversal_changes_donor():
    fluid = Properties().state_pt('gas', 1e5, 300.).as_dict()
    fluid['phase'] = 'gas'
    nodes = {'tank': gas_tank(5e5, .01),
             'boundary': dict(component='boundary', P=1e5, fluids={'gas': fluid})}
    with FluidNetwork(nodes, {'line': loss('tank', 'boundary')}, fluid_properties=Properties()) as net:
        net.initialize(bcs={'boundary': lambda t: {'P': 1e5 + t * 2e5}})
        result = net.update(3.)
        assert result['mdot']['line'] < 0 and net.directions['line'] == -1
        assert any(e['name'] == 'reverse' for e in result['events'])
        assert result['branch']['line']['flows']['main']['fluid']['T'] == 300.


def test_regulator_on_wet_tank_preserves_target_during_liquid_drain():
    nodes = {'supply': gas_tank(2e6), 'tank': wet_tank(mass=2., pressure=3e5),
             'ambient': dict(component='boundary', P=1e5)}
    branches = {'reg': dict(component='regulator', **{'from': 'supply', 'to': 'tank'},
                           to_port='ullage', CdA=5e-5, target_pressure=3e5),
                'drain': loss('tank', 'ambient', from_port='liquid')}
    with FluidNetwork(nodes, branches, fluid_properties=Properties()) as net:
        net.initialize()
        result = net.update(1.)
        assert result['node']['tank']['P'] == pytest.approx(3e5, abs=.2)
        assert result['node']['tank']['m_liq'] < 2.
        assert result['mdot']['reg'] > 0


@pytest.mark.parametrize('pressure', [2.9e5, 3.1e5])
def test_regulator_enters_regulation_from_either_side(pressure):
    nodes = {'supply': gas_tank(2e6), 'tank': gas_tank(pressure, .01),
             'ambient': dict(component='boundary', P=1e5)}
    branches = {'reg': dict(component='regulator', **{'from': 'supply', 'to': 'tank'},
                           CdA=5e-5, target_pressure=3e5), 'drain': loss('tank', 'ambient')}
    with FluidNetwork(nodes, branches, fluid_properties=Properties()) as net:
        result = net.update(1.)
        assert result['node']['tank']['P'] == pytest.approx(3e5, abs=.2)
        assert any(e['name'] == 'regulate' for e in result['events'])


def test_condensation_and_evaporation_preserve_inventory_and_energy():
    from FluidTables.PropertyModels import SaturationProperties
    class Saturating(Properties):
        def supports_saturation(self, fluid):
            return True
        def saturation_bounds(self, fluid):
            return (1., 1e7)
        def saturation_at_p(self, fluid, pressure):
            liquid = PureFluidProperties(pressure, 300., 1000., 150000. + pressure / 1000.,
                                         150000., 300., 1.4, 1e-5, .03, 500., 1/300.)
            return SaturationProperties(pressure, 300., liquid, self.state_pt(fluid, pressure, 300.))
    tank = gas_tank(3e5, .01)
    tank['state0']['T'] = 310.
    tank['state0']['P'] *= 310 / 300
    tank['state0']['U'] *= 310 / 300
    with FluidNetwork({'tank': tank}, {}, fluid_properties=Saturating()) as net:
        initial = net.initialize(heat_rate={'tank': {'gas': -100.}})
        cooled = net.update(3.)
        assert net.nodes['tank'].mode == 'saturated'
        assert cooled['node']['tank']['U'] == pytest.approx(initial['node']['tank']['U'] - 300., abs=.001)
        heated = net.update(5., heat_rate={'tank': {'gas': 100.}})
        assert net.nodes['tank'].mode == 'gas'
        assert heated['node']['tank']['m'] == initial['node']['tank']['m']
        assert heated['node']['tank']['U'] == pytest.approx(initial['node']['tank']['U'] + 200., abs=.001)
        assert [e['name'] for e in net.events] == ['condense', 'evaporate']


@pytest.mark.parametrize('tables', [False, True])
def test_real_oxygen_dryout_continues_nitrogen_flow(tables):
    from pathlib import Path
    from FluidTables.PropertyModels import CoolPropPropertySource, TablePureFluidPropertySource
    props = TablePureFluidPropertySource(Path(__file__).resolve().parents[1] / 'FluidTables/sizer_lookups.h5',
                                        {'Oxygen': 'oxygen_pt', 'Nitrogen': 'nitrogen_pt'}) if tables else CoolPropPropertySource()
    liquid = props.state_pt('Oxygen', 1e6, 95.)
    gas = props.state_pt('Nitrogen', 1e6, 300.)
    ml, mg = .05, (.01 - .05 / liquid.rho) * gas.rho
    nodes = {'tank': dict(component='propellant_tank', liquid_fluid='Oxygen', gas_fluid='Nitrogen',
                         geometry=Geometry(), state0=dict(P=1e6, T=95., gas_T=300., m_liq=ml,
                         U_liq=ml * liquid.u, m_ull=mg, U_ull=mg * gas.u)),
             'ambient': dict(component='boundary', P=1e5)}
    with FluidNetwork(nodes, {'line': loss('tank', 'ambient', from_port='liquid')},
                      fluid_properties=props, tolerances={'max_step': .01}) as net:
        result = net.update(.5)
        assert net.nodes['tank'].mode == 'gas'
        assert result['mdot']['line'] > 0
        assert result['numerical_remainders'][0]['fluid'] == 'Oxygen'
        assert list(result['node']['tank']['fluids']) == ['Nitrogen']


def test_failed_dryout_restart_restores_modes_and_event_history():
    with propulsion() as net:
        net.initialize()
        y, dy = net.y.copy(), net.ydot.copy()
        old_prepare = net._new_session
        def fail_restart(*, seed=False):
            if seed:
                raise RuntimeError('intentional restart failure')
            return old_prepare(seed=seed)
        net._new_session = fail_restart
        with pytest.raises(RuntimeError, match='intentional restart failure'):
            net.update(3.)
        np.testing.assert_array_equal(net.y, y)
        np.testing.assert_array_equal(net.ydot, dy)
        assert net.nodes['ox'].mode == net.nodes['fuel'].mode == 'two_phase'
        assert net.nodes['chamber'].mode == 'combusting'
        assert net.events == [] and net.remainders == [] and net.time == 0.
        assert net.session is None


def test_residual_is_pure_and_outputs_are_detached():
    with propulsion() as net:
        output = net.initialize()
        original = net._checkpoint()
        first, second = np.empty_like(net.y), np.empty_like(net.y)
        net.residual(net.time, net.y, net.ydot, first)
        perturbed = net.y.copy()
        perturbed[net.variable_index['ox.P']] *= 1.001
        net.residual(net.time, perturbed, net.ydot, second)
        net.residual(net.time, net.y, net.ydot, second)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(net.y, original['y'])
        assert net.events == original['events']
        output['node']['ox']['P'] = -1
        output['state'].nodes['ox'].trial_values['P'] = -1
        assert net.state.nodes['ox']['P'] > 0


def test_constraint_monitor_receives_network_state():
    observed = []
    def monitor(state):
        pressure = state.nodes['tank']['P']
        observed.append(pressure)
        return {'pressure_margin': pressure - 1e5}
    with blowdown(constraint_monitor=monitor) as net:
        result = net.update(1.)
        assert observed
        assert result['constraints']['pressure_margin'] == pytest.approx(result['node']['tank']['P'] - 1e5)
        assert result['constraint_times']['pressure_margin'] == 1.


def test_full_regulated_pumpfed_network_events_and_refinement():
    with propulsion(pump=True) as template:
        nodes, branches = template.node_definitions, template.branch_definitions
    nodes['copv'] = gas_tank(3e6, .03)
    for tank in ('ox', 'fuel'):
        branches[f'{tank}_reg'] = dict(component='regulator', **{'from': 'copv', 'to': tank},
                                      to_port='ullage', CdA=2e-5, target_pressure=1e6)
    results = []
    for rtol, step in ((1e-7, .05), (1e-8, .02)):
        with FluidNetwork(nodes, branches, fluid_properties=Properties(), combustion_properties=Combustion(),
                          tolerances={'rtol': rtol, 'max_step': step}) as net:
            initial = net.initialize()
            prediction = net.update(2., commit=False)
            assert net.time == 0 and net.nodes['ox'].mode == 'two_phase' and net.events == []
            result = net.update(2.)
            assert len([e for e in result['events'] if e['name'] == 'dryout']) == 2
            assert any(e['name'] == 'shutdown' for e in result['events'])
            assert result['node']['chamber']['mode'] == 'shutdown'
            assert result['mdot']['nozzle'] > 0
            assert result['node']['copv']['m'] < initial['node']['copv']['m']
            assert result['node']['ox']['P'] == pytest.approx(1e6, rel=2e-6)
            assert result['node']['fuel']['P'] == pytest.approx(1e6, rel=2e-6)
            assert result['mdot']['nozzle'] == pytest.approx(prediction['mdot']['nozzle'], rel=1e-5)
            results.append(result)
    times = [[e['time_s'] for e in r['events'] if e['name'] == 'dryout'] for r in results]
    np.testing.assert_allclose(times[0], times[1], rtol=2e-5, atol=1e-6)
    assert results[0]['mdot']['nozzle'] == pytest.approx(results[1]['mdot']['nozzle'], rel=2e-5)


def test_junction_routes_current_donor_with_algebraic_only_network():
    gas = Properties().state_pt('gas', 5e5, 300.).as_dict()
    gas['phase'] = 'gas'
    nodes = {'supply': dict(component='boundary', P=5e5, fluids={'gas': gas}),
             'junction': dict(component='junction', fluid='gas', phase='gas', state0=dict(P=3e5,T=300.)),
             'ambient': dict(component='boundary', P=1e5)}
    branches = {'inlet': loss('supply', 'junction'), 'outlet': loss('junction', 'ambient')}
    with FluidNetwork(nodes, branches, fluid_properties=Properties()) as net:
        result = net.update(1.)
        assert not net.differential.any()
        assert result['mdot']['inlet'] == pytest.approx(result['mdot']['outlet'], rel=1e-7)
        assert 1e5 < result['node']['junction']['P'] < 5e5
        assert result['node']['junction']['fluids']['gas']['T'] == pytest.approx(300.)


def test_initially_out_of_band_valve_settles_before_integration():
    nodes = {'supply': gas_tank(2e6), 'tank': gas_tank(4e5, .01)}
    branches = {'valve': dict(component='bang_bang_valve', **{'from': 'supply', 'to': 'tank'},
                             CdA=3e-5, target_pressure=3e5, pressure_band=2e4, initially_open=True)}
    with FluidNetwork(nodes, branches, fluid_properties=Properties()) as net:
        result = net.initialize()
        assert not net.branches['valve'].is_open and result['mdot']['valve'] == 0
        assert net.events[0]['time_s'] == 0
        before = deepcopy(result['node'])
        result = net.update(1.)
        for key in nodes:
            assert result['node'][key]['m'] == before[key]['m']
            assert result['node'][key]['U'] == before[key]['U']


def test_stop_at_shutdown_freezes_at_event_and_preview_is_isolated():
    with propulsion(stop_at_shutdown=True, tolerances={'max_step': .05}) as net:
        initial=net.initialize()
        trial=net.update(3.,commit=False)
        assert trial['frozen'] and net.time==0 and not net.frozen
        stopped=net.update(3.)
        assert stopped['frozen'] and stopped['stop_reason']=='engine_shutdown'
        assert all(flow==0 for flow in stopped['mdot'].values())
        assert stopped['node']['fuel']['mass']>0
        assert net.session is None
        later=net.update(2.)
        assert later['td_state']==stopped['td_state']
        assert later['time']==5.


def test_bound_monitor_preview_and_failed_update_restore_state():
    class Owner:
        fail=False
        def monitor(self,state):
            if self.fail and state.nodes['tank']['P'] < 899000:
                raise RuntimeError('monitor failure')
            return {'pressure':state.nodes['tank']['P']-1e5}
    owner=Owner()
    owner.network=blowdown(constraint_monitor=owner.monitor)
    with owner.network as net:
        net.initialize()
        before=net.y.copy()
        preview=net.update(.1,commit=False)
        np.testing.assert_array_equal(net.y,before)
        owner.fail=True
        with pytest.raises(RuntimeError,match='monitor failure'):
            net.update(1.)
        np.testing.assert_array_equal(net.y,before)
        assert net.time==0 and net.events==[] and net.session is None
        owner.fail=False
        assert net.update(.1)['node']['tank']['P']==pytest.approx(preview['node']['tank']['P'])


def test_triple_point_stops_at_temperature_and_warm_low_pressure_gas_does_not_stop():
    from FluidTables.PropertyModels import SaturationProperties
    class TripleProperties(Properties):
        def supports_saturation(self,fluid): return True
        def saturation_bounds(self,fluid): return (1e5,1e9)
        def saturation_at_p(self,fluid,pressure):
            state=self.state_pt(fluid,pressure,200.)
            return SaturationProperties(pressure,200.,state,state)
    with FluidNetwork({'tank':gas_tank(9e5,.1)}, {}, fluid_properties=TripleProperties(),
                      stop_at_triple_point=True) as net:
        first=net.initialize(heat_rate={'tank':{'gas':-1e5}})
        out=net.update(2.)
        assert out['frozen'] and out['stop_reason']=='triple_point'
        assert out['node']['tank']['T']==pytest.approx(200.,abs=.001)
        event=next(e for e in out['events'] if e['name'].startswith('triple_point:'))
        assert out['node']['tank']['U']==pytest.approx(first['node']['tank']['U']-event['time_s']*1e5,abs=.01)
        assert net.update(1.)['td_state']==out['td_state']
    with FluidNetwork({'tank':gas_tank(5e4,.1)}, {}, fluid_properties=TripleProperties(),
                      stop_at_triple_point=True) as net:
        assert not net.update(.1)['frozen']


def test_boundary_changes_reuse_allocations_but_reinitialize_history(monkeypatch):
    from Fluids.Sundials.ida_session import IdaSession
    original, restarts = IdaSession.restart, []
    def restart(self, *args, **kwargs):
        restarts.append(self.time)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(IdaSession, 'restart', restart)
    with blowdown() as net, blowdown() as rebuilt:
        net.update(.1,bcs={'ambient':{'P':1e5}},heat_rate={})
        rebuilt.update(.1,bcs={'ambient':{'P':1e5}},heat_rate={})
        session=net.session
        net.update(.1,bcs={'ambient':{'P':1e5}},heat_rate={})
        rebuilt.update(.1,bcs={'ambient':{'P':1e5}},heat_rate={})
        assert net.session is session and not restarts
        net.update(.1,bcs={'ambient':{'P':1.1e5}},heat_rate={})
        rebuilt.close()
        rebuilt.update(.1,bcs={'ambient':{'P':1.1e5}},heat_rate={})
        assert net.session is session and len(restarts) == 1
        np.testing.assert_allclose(net.y, rebuilt.y, rtol=2e-8, atol=1e-9)
        np.testing.assert_allclose(net.ydot, rebuilt.ydot, rtol=2e-8, atol=1e-9)
        # Adding hydrostatic dependencies changes the Jacobian structure.
        net.update(.1, axial_specific_force=10.)
        assert net.session is not session and session.owner is None


@pytest.mark.parametrize('persistent,error_type,attempts', [
    (False, 'residual', 2), (True, 'residual', 3), (True, 'other', 1)])
def test_initialization_retries_only_residual_failure_and_restores_state(monkeypatch, persistent, error_type, attempts):
    from errors import ResidualAcceptanceError
    original, calls = FluidNetwork._accept, []
    error = ResidualAcceptanceError if error_type == 'residual' else ValueError
    def accept(net):
        original(net)
        calls.append(net.effective_rtol)
        if persistent or len(calls) == 1:
            net.events.append({'name': 'rejected'})
            raise error('injected initialization rejection')
    monkeypatch.setattr(FluidNetwork, '_accept', accept)
    with blowdown(tolerances={'rtol': 1e-4}) as net:
        if persistent:
            with pytest.raises(error, match='injected'):
                net.initialize()
            assert len(calls) == attempts
            assert net.y is None and net.session is None and net.time == 0
            assert net.effective_rtol == 1e-4 and not net.retry_diagnostics
        else:
            net.initialize()
            assert net.time == 0 and net.effective_rtol == pytest.approx(1e-5)
            assert len(net.retry_diagnostics) == 1
            assert net.retry_diagnostics[0]['initialization_retry']
        assert not net.events


def test_accepted_residual_retry_restores_events_and_retains_tighter_integration(monkeypatch):
    from errors import ResidualAcceptanceError
    original = FluidNetwork._accept
    failures = []
    def accept(net):
        original(net)
        if net.time > 0 and len(failures) < 2:
            failures.append((net.time, net.effective_rtol, net.effective_max_step))
            # These trial-only diagnostics must be rolled back too.
            net.constraints['rejected_attempt'] = -100.
            net.events.append(dict(time_s=net.time, kind='node', component='tank', name='rejected'))
            raise ResidualAcceptanceError('injected closure failure')
    with blowdown(tolerances={'rtol': 1e-4, 'max_step': 1.}) as net:
        net.initialize()
        monkeypatch.setattr(FluidNetwork, '_accept', accept)
        out = net.update(1.)
        assert net.time == 1. and len(failures) == 2
        assert 'rejected_attempt' not in out['constraints'] and out['events'] == ()
        assert net.options['rtol'] == 1e-4
        assert net.effective_rtol == pytest.approx(1e-6)
        assert net.effective_max_step == pytest.approx(.05)
        assert len(net.retry_diagnostics) == 2
        net.update(.1)
        assert net.effective_rtol == pytest.approx(1e-6)


def test_accepted_residual_retry_is_bounded_and_rolls_back(monkeypatch):
    from errors import ResidualAcceptanceError
    attempts = []
    original = FluidNetwork._accept
    def accept(net):
        original(net)
        if net.time > 0:
            attempts.append(net.time)
            raise ResidualAcceptanceError('persistent closure failure')
    with blowdown(tolerances={'rtol': 1e-9}) as net:
        net.initialize()
        before = net._checkpoint()
        monkeypatch.setattr(FluidNetwork, '_accept', accept)
        with pytest.raises(ResidualAcceptanceError, match='persistent closure failure') as failure:
            net.update(1.)
        assert len(failure.value.attempts) == 3
        assert len(attempts) == 3 and net.time == 0
        np.testing.assert_array_equal(net.y, before['y'])
        assert net.effective_rtol == 1e-9 and net.events == [] and net.session is None


def test_checkpoint_can_be_restored_repeatedly_across_shutdown():
    with propulsion(stop_at_shutdown=True) as net:
        net.initialize()
        checkpoint = net._checkpoint()
        for _ in range(2):
            out = net.update(3.)
            shutdown = next(e for e in out['events'] if e['name'] == 'shutdown')
            assert shutdown['before']['node']['chamber']['mode'] == 'combusting'
            assert shutdown['before']['mdot']['nozzle'] > 0
            net.restore(checkpoint)
            assert not net.frozen and net.events == [] and net.time == 0
            assert net.nodes['ox'].mode == 'two_phase'


@pytest.mark.parametrize('reverse_order', [False, True])
def test_algebraic_mixer_conserves_connected_enthalpy_flux(reverse_order):
    props = Properties()
    def supply(temperature):
        return dict(component='boundary',P=5e5,fluids={
            'gas':dict(phase='gas',**props.state_pt('gas',5e5,temperature).as_dict())})
    nodes = {'cold':supply(300.),'hot':supply(600.),
             'mix':dict(component='junction',fluid='gas',phase='gas',state0=dict(P=3e5,T=350.)),
             'ambient':dict(component='boundary',P=1e5)}
    branches = {'cold_in':loss('cold','mix'), 'hot_in':loss('hot','mix'),
                'out':loss('mix','ambient',2e-5)}
    if reverse_order:
        branches = dict(reversed(list(branches.items())))
    with FluidNetwork(nodes,branches,fluid_properties=props,
                      tolerances={'verify_jacobian':True}) as net:
        result = net.initialize()
        cold,hot,out = (result['mdot'][k] for k in ('cold_in','hot_in','out'))
        temperature = result['node']['mix']['T']
        assert 300 < temperature < 600
        assert out == pytest.approx(cold+hot,rel=1e-7)
        assert temperature == pytest.approx((cold*300+hot*600)/(cold+hot),rel=1e-7)
        assert net.session.structured_verified
        later = net.update(.1)
        assert later['node']['mix']['T'] == pytest.approx(temperature,rel=1e-7)


@pytest.mark.parametrize('enabled', [False, True])
def test_junction_zero_throughput_is_initialized_explicitly(enabled):
    props = Properties()
    gas = dict(phase='gas',**props.state_pt('gas',2e5,300.).as_dict())
    nodes = {'left':dict(component='boundary',P=2e5,fluids={'gas':gas}),
             'mix':dict(component='junction',fluid='gas',phase='gas',state0=dict(P=2e5,T=310.)),
             'right':dict(component='boundary',P=2e5,fluids={'gas':gas})}
    branches = {'in':loss('left','mix',enabled=enabled,mdot0=0.),
                'out':loss('mix','right',enabled=enabled,mdot0=0.)}
    with FluidNetwork(nodes,branches,fluid_properties=props) as net:
        result = net.initialize()
        assert result['mdot']['in'] == pytest.approx(0.,abs=1e-10)
        assert result['mdot']['out'] == pytest.approx(0.,abs=1e-10)
        assert result['node']['mix']['T'] == pytest.approx(310.)


def test_junction_reversal_reinitializes_energy_balance():
    props = Properties()
    def supply(temperature):
        return dict(component='boundary',P=5e5,fluids={
            'gas':dict(phase='gas',**props.state_pt('gas',5e5,temperature).as_dict())})
    nodes = {'left':supply(300.),'right':supply(500.),
             'mix':dict(component='junction',fluid='gas',phase='gas',state0=dict(P=4e5,T=350.))}
    branches = {'in':loss('left','mix'), 'out':loss('mix','right')}
    boundaries = {'left':lambda t:{'P':5e5-2e5*t}, 'right':{'P':4e5}}
    with FluidNetwork(nodes,branches,fluid_properties=props,
                      tolerances={'max_step':.02,'verify_jacobian':True}) as net:
        first = net.initialize(bcs=boundaries)
        assert first['node']['mix']['T'] == pytest.approx(300.)
        result = net.update(1.,bcs=boundaries)
        assert result['mdot']['in'] < 0 and result['mdot']['out'] < 0
        assert result['node']['mix']['T'] == pytest.approx(500.)
        assert any(e['name']=='reverse' for e in result['events'])
