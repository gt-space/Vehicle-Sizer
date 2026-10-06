"""Valve laws and localized events exercise the same component event contract."""
from math import pi, sin

import pytest

from Fluids import FluidNetwork
from Fluids.FluidBranch import LossComponent, ReliefValveComponent
from Fluids.FluidNode import VolumeComponent
from Fluids.events import Event
from test_fluid_network import Properties, gas_tank, loss
from test_prop_system import config, build
from Fluids.helpers.templates import load_template


def switch(**overrides):
    return dict(component='switch_valve', **{'from': 'tank', 'to': 'ambient'},
                CdA_before=1e-6, CdA_after=2e-6, sense_node='sensor',
                switch_pressure=2.5e5, **overrides)


def network(branch, **options):
    return FluidNetwork({'tank': gas_tank(), 'ambient': dict(component='boundary', P=1e5),
                         'sensor': dict(component='boundary', P=2e5)},
                        {'valve': branch}, fluid_properties=Properties(), **options)


@pytest.mark.parametrize('latched', [True, False])
def test_switch_localization_latching_hysteresis_and_preview(latched):
    definition = switch(latched=latched, reset_pressure=1.5e5)
    with network(definition, tolerances={'max_step': .2}) as net:
        net.initialize(bcs={'sensor': lambda t: {'P': 2e5 + 1e5 * sin(t)}})
        saved = net._checkpoint()
        preview = net.update(13., commit=False)
        assert net.time == 0 and not net.branches['valve'].is_switched and not net.events
        result = net.update(13.)
        events = [e for e in result['events'] if e['name'] == 'switch']
        expected = [pi / 6] if latched else [pi / 6, 7*pi / 6, 13*pi / 6, 19*pi / 6]
        assert [e['time_s'] for e in events] == pytest.approx(expected, abs=1e-6)
        assert result['node']['tank']['m'] == pytest.approx(preview['node']['tank']['m'])
        assert result['branch']['valve']['effective_CdA'] == (2e-6 if latched else 1e-6)
        net.restore(saved)
        assert not net.branches['valve'].is_switched
        assert net.update(13.)['node']['tank']['m'] == pytest.approx(result['node']['tank']['m'])


@pytest.mark.parametrize('direction,pressure', [('rising', 3e5), ('falling', 2e5)])
def test_switch_initial_guard_disabled_and_zero_area(direction, pressure):
    definition = switch(switch_direction=direction)
    definition['CdA_before'] = 0.
    with network(definition) as net:
        state = net.initialize(bcs={'sensor': {'P': pressure}})
        assert state['branch']['valve']['is_switched']
        assert state['mdot']['valve'] > 0
        assert state['events'][0]['time_s'] == 0
    definition['enabled'] = False
    with network(definition) as net:
        state = net.initialize(bcs={'sensor': {'P': pressure}})
        assert not state['events'] and state['mdot']['valve'] == 0


def test_relief_repeated_cycles_and_actual_differential_thresholds():
    nodes = {'supply': gas_tank(3e6, 10.), 'tank': gas_tank(2e5, .01),
             'ambient': dict(component='boundary', P=1e5)}
    branches = {'feed': loss('supply', 'tank', 1e-6),
                'valve': dict(component='relief_valve', **{'from': 'tank', 'to': 'ambient'},
                              CdA=3e-5, open_dP=3e5, close_dP=2e5)}
    with FluidNetwork(nodes, branches, fluid_properties=Properties()) as net:
        net.initialize()
        result = net.update(10.)
        events = [e for e in result['events'] if e['component'] == 'valve']
        assert len(events) >= 4
        assert [e['is_open'] for e in events] == [i % 2 == 0 for i in range(len(events))]
        # Replay each localized endpoint; accepted pressure must match its guard.
        times = [e['time_s'] for e in events]
    with FluidNetwork(nodes, branches, fluid_properties=Properties()) as net:
        net.initialize()
        for i, time in enumerate(times):
            state = net.update(time - net.time)
            assert state['node']['tank']['P'] - 1e5 == pytest.approx(3e5 if i % 2 == 0 else 2e5, rel=2e-5)


def test_relief_initial_opening_reseating_and_no_reverse_donor():
    definition = dict(component='relief_valve', **{'from': 'tank', 'to': 'ambient'},
                      CdA=1e-5, open_dP=5e5, close_dP=3e5)
    with network(definition) as net:
        initial = net.initialize()
        assert initial['branch']['valve']['is_open']
        final = net.update(100.)
        assert not final['branch']['valve']['is_open']
        assert final['node']['tank']['P'] == pytest.approx(4e5, rel=1e-6)
        # Ambient deliberately has no fluid state: it must never become donor.
        state = net.update(1., bcs={'ambient': {'P': 2e6}})
        assert state['mdot']['valve'] == 0
        assert net.branches['valve'].direction == 1


@pytest.mark.parametrize('phase', ['gas', 'liquid'])
def test_relief_reuses_loss_law_and_blocks_backflow(phase):
    from test_fluid_branches import nodes, fluid
    definition = dict(**{'from': 'up', 'to': 'down'}, CdA=1e-5, open_dP=1e5,
                      close_dP=5e4, initially_open=True)
    valve = ReliefValveComponent('v', definition, phase=phase)
    reference = LossComponent('loss', definition, phase=phase)
    fluids = {'source': fluid(phase)}
    for up, down in [(5e5, 1e5), (2e5, 1.9e5), (1e5, 2e5)]:
        states = nodes(up=up, down=down)
        trial = valve.evaluate({'mdot': 0.}, states, fluids, {'main': 1})
        assert valve.mass_flow(trial, states) == max(0., reference.mass_flow(trial, states))


def test_arbitrary_time_and_inventory_rule_needs_no_dispatcher_changes():
    class CustomLoss(LossComponent):
        def events(self, context):
            # A completely new event name and guard: armed by local mode,
            # conditional on another component, localized using time.
            if self.enabled and context.values['tank']['m'] > .1:
                return {'scheduled_isolation': Event(1.25 - context.time, at_zero=True)}
            return {}

        def apply_event(self, name, values, time):
            assert name == 'scheduled_isolation'
            self.set_enabled(False)
            return {'isolated': True}

    with network(loss('tank', 'ambient')) as net:
        net.branches['valve'] = CustomLoss('valve', loss('tank', 'ambient'))
        net.initialize()
        result = net.update(3.)
        assert result['events'][0]['time_s'] == pytest.approx(1.25)
        assert result['events'][0]['isolated']
        assert result['mdot']['valve'] == 0


def test_generic_node_activation_and_terminal_event():
    class TimedVolume(VolumeComponent):
        armed = False

        def events(self, context):
            if self.armed:
                return {'finish': Event(2. - context.time, terminal='custom_stop')}
            return {'arm': Event(.5 - context.time, at_zero=True)}

        def apply_event(self, name, values, time):
            assert name == 'arm'
            self.armed = True
            return {}

    with network(loss('tank', 'ambient')) as net:
        net.nodes['tank'] = TimedVolume('tank', gas_tank(), fluid_properties=Properties())
        net.initialize()
        result = net.update(3.)
        assert [e['name'] for e in result['events']] == ['arm', 'finish']
        assert [e['time_s'] for e in result['events']] == pytest.approx([.5, 2.])
        assert result['frozen'] and result['stop_reason'] == 'custom_stop'


def test_coincident_valves_and_failed_transition_rollback(monkeypatch):
    definition = switch()
    with network(definition) as net:
        net.branches['second'] = type(net.branches['valve'])('second', definition)
        net.connections['tank'].append((-1, 'second'))
        net.connections['ambient'].append((1, 'second'))
        net.initialize(bcs={'sensor': lambda t: {'P': 2e5 + t*1e5}})
        initial_mass = net.state.nodes['tank']['m']
        original = type(net.branches['valve']).apply_event

        def fail_after_transition(self, *args):
            original(self, *args)
            raise RuntimeError('injected transition failure')

        with monkeypatch.context() as patch:
            patch.setattr(type(net.branches['valve']), 'apply_event', fail_after_transition)
            with pytest.raises(RuntimeError, match='injected transition failure'):
                net.update(1.)
        assert net.time == 0 and not net.events
        assert all(not b.is_switched for b in net.branches.values())
        assert net.state.nodes['tank']['m'] == initial_mass
        result = net.update(1.)
        assert len(result['events']) == 2
        assert [e['time_s'] for e in result['events']] == pytest.approx([.5, .5])


@pytest.mark.parametrize('kind', ['switch_valve', 'relief_valve'])
def test_propsystem_uses_config_areas_without_sizing(kind):
    cfg = config()
    template = load_template(cfg)
    if kind == 'switch_valve':
        branch = template['branches']['OX_TANK_INJ']
        branch.update(component=kind, CdA_before=1.23e-5, CdA_after=4.56e-5,
                      switch_pressure=2e6)
        branch_id = 'OX_TANK_INJ'
    else:
        branch_id = 'vent'
        template['branches'][branch_id] = dict(component=kind, circuit='pressurant',
            **{'from': 'ox_ullage', 'to': 'ambient'}, from_port='ullage',
            CdA=1.23e-5, open_dP=4e6, close_dP=3e6)
    cfg['prop_system']['template'] = template
    with build(cfg) as prop:
        valve = prop.network.branches[branch_id]
        assert valve.cda == 1.23e-5
        assert 'design_mdot' not in valve.parameters
        if kind == 'switch_valve':
            valve.apply_event('switch', {}, 0.)
            assert valve.cda == 4.56e-5


@pytest.mark.parametrize('field,value', [('CdA_before', None), ('CdA_after', -1),
                                        ('switch_direction', 'invalid'), ('initially_switched', 1),
                                        ('latched', 1), ('switch_pressure', 0), ('sense_node', 'missing')])
def test_invalid_switch_inputs_fail_before_integration(field, value):
    definition = switch()
    definition[field] = value
    with pytest.raises(ValueError):
        network(definition)


@pytest.mark.parametrize('overrides', [dict(CdA=None), dict(CdA=-1), dict(close_dP=4e5),
                                     dict(initially_open=1), dict(direction=-1)])
def test_invalid_relief_inputs_fail_before_integration(overrides):
    definition = dict(component='relief_valve', **{'from': 'tank', 'to': 'ambient'},
                      CdA=1e-5, open_dP=3e5, close_dP=2e5)
    definition.update(overrides)
    with pytest.raises(ValueError):
        network(definition)
