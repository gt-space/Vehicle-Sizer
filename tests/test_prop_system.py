"""The same declarative templates serve sizing, IDAS and the flight adapter."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from Fluids.design import initial_conditions, tank_design_pressure
from Fluids.templates import load_template
from Fluids.PropSystem import PropSystem
from FluidTables.PropertyModels import CEAPropertySource, CoolPropPropertySource, TablePureFluidPropertySource
import propulsion_fixtures as fixtures
from propulsion_fixtures import GeometrySource, TankGeometry, GasGeometry, FakeCEA


def config(feed='pressure_fed', control='regulator'):
    cfg = fixtures.config()
    prop = cfg['prop_system']
    prop['template'] = f'Configs/templates/{feed}_{control}.yaml'
    prop['regulator'] = {leg + '_REGULATOR': dict(capacity_factor=2., collapse_factor=1.2, min_temperature=220.)
                         for leg in ('OX', 'FUEL')}
    if feed == 'pump_fed':
        prop['pumps'] = {key: dict(drive='electric', pressure_rise_pa=1e6, gas_CdA=1e-5,
                                  efficiency=.7, max_power_kw=100.) for key in ('oxidizer_pump','fuel_pump')}
        for leg in ('ox','fuel'):
            prop[leg+'_pumpin_tank_dp'] = 2e5
            prop[leg+'_inj_pumpout_dp'] = 2e5
    for state in prop['initial_conditions'].values():
        for key in ('P', 'm', 'U', 'm_liq', 'U_liq', 'm_ull', 'U_ull'):
            state.pop(key, None)
    if control == 'blowdown':
        del prop['initial_conditions']['press_tank']
        del cfg['tanks']['press_tank']
    return cfg


def build(cfg, mass=.2, tabled=False):
    props = (TablePureFluidPropertySource(Path(__file__).resolve().parents[1] / 'FluidTables/sizer_lookups.h5',
             {'Oxygen':'oxygen_pt','Nitrogen':{'pt':'nitrogen_pt','saturation':'nitrogen_saturation'},
              'n-Dodecane':'ndodecane_pt'}) if tabled else CoolPropPropertySource())
    states = initial_conditions(cfg)
    tanks = {}
    for key,state in states.items():
        if 'gas_fluid' in state:
            liquid_mass = mass if key == 'ox_tank' else 2 * mass
            rho = props.state_pt(state['fluid'], state['P'], state['T']).rho
            tanks[key] = GeometrySource(TankGeometry(liquid_mass/rho + .01), prop_mass=liquid_mass)
        else:
            tanks[key] = GeometrySource(GasGeometry(.05))
    return PropSystem(cfg,tanks,props,CEAPropertySource(FakeCEA()))


@pytest.mark.parametrize('feed', ['pressure_fed','pump_fed'])
@pytest.mark.parametrize('control', ['blowdown','regulator','bang_bang'])
def test_templates_size_without_solver_and_preserve_design_physics(feed, control):
    cfg = config(feed,control)
    before = deepcopy(cfg)
    with build(cfg) as prop:
        assert prop.throat_area == pytest.approx(12000 / (2e6*1.5))
        assert prop.mdot_ox/prop.mdot_fuel == 3
        assert prop.network.y is None
        assert set(prop.network.branches) == set(load_template(cfg)['branches'])
        assert len(prop.pump_sizing) == (2 if feed == 'pump_fed' else 0)
        assert tank_design_pressure(cfg,'ox_tank') == pytest.approx(1.8e6 if feed=='pump_fed' else 2.6e6)
    assert cfg == before


def test_template_is_required_and_topology_not_selected_by_feed_flags():
    cfg=config()
    cfg['prop_system']['feed_type']='anything'
    cfg['prop_system']['pressurization']='anything'
    with build(cfg) as prop:
        assert 'OX_REGULATOR' in prop.network.branches
    del cfg['prop_system']['template']
    with pytest.raises(ValueError,match='template'):
        load_template(cfg)


@pytest.mark.parametrize('control,expected', [('regulator', 1e-4), ('bang_bang', 1e-5), ('blowdown', 1e-7)])
def test_template_sets_default_rtol_and_config_overrides(control, expected):
    cfg = config('pump_fed', control)
    # Template wiring, not these redundant labels, selects the tolerance.
    cfg['prop_system']['pressurization'] = 'unused_label'
    with build(cfg) as prop:
        assert prop.network.options['rtol'] == expected
        assert prop.network.options['jacobian'] == 'colored'
    cfg['advanced'] = {'fluid_network': {'rtol': 2e-6, 'jacobian': 'dense'}}
    with build(cfg) as prop:
        assert prop.network.options['rtol'] == 2e-6
        assert prop.network.options['jacobian'] == 'dense'


def test_custom_template_renames_nodes_and_adds_a_loss_without_code_changes():
    cfg=config('pump_fed')
    template=load_template(cfg)
    template['nodes']['extra_junction']={'component':'junction','P0':2.8e6, 'fluid':'Oxygen', 'phase':'liquid', 'state0':{'P':2.8e6,'T':90.}}
    template['nodes']['ox_pump_out']['P0']=3e6
    template['nodes']['ox_pump_in']['P0']=2e6
    template['nodes']['ox_ullage']['P0']=2.2e6
    template['branches']['extra_loss']={'component':'loss','circuit':'oxidizer','from':'ox_pump_out','to':'extra_junction'}
    template['branches']['OX_PUMP_INJ']['from']='extra_junction'
    renames={key:'custom_'+key for key in template['nodes']}
    template['nodes']={renames[key]:node for key,node in template['nodes'].items()}
    for node in template['nodes'].values():
        if 'ambient_node' in node: node['ambient_node']=renames[node['ambient_node']]
    for branch in template['branches'].values():
        for end in ('from','to'): branch[end]=renames[branch[end]]
    cfg['prop_system']['template']=template
    with build(cfg) as prop:
        assert prop.chamber_id=='custom_thrust_chamber'
        assert 'extra_loss' in prop.network.branches
        assert prop.branch_definitions['extra_loss']['CdA']>0


def test_template_pressure_cycle_and_wiring_override_are_rejected():
    cfg=config()
    template=load_template(cfg)
    cfg['prop_system']['template']=template
    template['nodes']['ox_ullage']['P0']={'node':'ox_ullage'}
    with pytest.raises(ValueError,match='Cyclic'):
        load_template(cfg)
    template['nodes']['ox_ullage']['P0']=2.6e6
    template['branches']['OX_REGULATOR']['parameters']={'to':'ambient'}
    with pytest.raises(ValueError,match='wiring'):
        load_template(cfg)


@pytest.mark.parametrize('feed,control', [('pressure_fed','blowdown'), ('pressure_fed','bang_bang'), ('pump_fed','regulator')])
def test_actual_template_preview_commit_dryout_and_stopping(feed,control):
    pytest.importorskip('sundials4py')
    cfg=config(feed,control)
    cfg['simulation']={'fluid_solve_post_shutdown':False}
    cfg['advanced']={'fluid_network':{'max_step':.01}}
    with build(cfg,tabled=True) as prop:
        atm=SimpleNamespace(p=1e5)
        first=prop.update(None,atm,{})
        assert first.propulsion.mode=='combusting'
        y=prop.network.y.copy()
        trial=prop.update(.2,atm,{},commit=False)
        np.testing.assert_array_equal(y,prop.network.y)
        assert prop.network.time==0 and not prop.network.events
        actual=prop.update(.2,atm,{})
        assert actual.propulsion.mode=='shutdown'
        assert actual.propulsion.thrust==actual.propulsion.mdot_nozzle==0
        assert prop.network.frozen
        assert any(event['name']=='dryout' for event in actual.events)
        assert trial.propulsion.thrust==actual.propulsion.thrust
        for key in first.td_state:
            assert actual.td_state[key]['mass']==pytest.approx(trial.td_state[key]['mass'])
        later=prop.update(.3,atm,{})
        assert later.events==()
        assert later.event_counts==actual.event_counts
        assert later.td_state==actual.td_state
        assert prop.network.time==pytest.approx(.5)


def test_template_continues_passive_flow_after_shutdown():
    pytest.importorskip('sundials4py')
    cfg=config('pump_fed','regulator')
    cfg['advanced']={'fluid_network':{'max_step':.01}}
    with build(cfg,tabled=True) as prop:
        out=prop.update(.15,SimpleNamespace(p=1e5),{})
        assert out.propulsion.mode=='shutdown'
        assert out.propulsion.mdot_nozzle>0
        assert not prop.network.frozen
