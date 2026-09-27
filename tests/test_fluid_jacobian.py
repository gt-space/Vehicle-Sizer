"""Production coloring must preserve solutions through equation changes."""
import numpy as np
import pytest

pytest.importorskip('sundials4py')
from test_fluid_network import blowdown, propulsion
from test_prop_system import config, build
from types import SimpleNamespace


def test_colored_analytic_blowdown():
    with blowdown(tolerances={'jacobian': 'dense'}) as net:
        net.initialize()
        reference = net.update(10.)
    with blowdown(tolerances={'verify_jacobian': True}) as net:
        net.initialize()
        result = net.update(10.)
        assert net.session.structured_verified
        assert net.session.statistics()['structured_residuals'] > 0
        assert result['node']['tank']['P'] == pytest.approx(reference['node']['tank']['P'], rel=2e-6)


def test_colored_hydrostatic_ports():
    with propulsion(pump=True, tolerances={'verify_jacobian': True}) as net:
        net.update(None, axial_specific_force=20.)
        net.update(.01, axial_specific_force=20.)
        assert net.session.structured_verified
        # Changed acceleration rebuilds the pattern for hydrostatic dependencies.
        net.update(.01, axial_specific_force=40.)
        assert net.session.structured_verified


@pytest.mark.parametrize('simultaneous', [False, True])
def test_colored_dryout_shutdown_and_passive_flow(simultaneous):
    with propulsion(pump=True, simultaneous=simultaneous,
                    tolerances={'max_step': .05, 'verify_jacobian': True}) as net:
        net.initialize()
        result = net.update(3.)
        assert any(e['name'] == 'dryout' for e in result['events'])
        assert result['node']['chamber']['mode'] == 'shutdown'
        assert result['mdot']['nozzle'] > 0
        assert net.session.structured_verified
        assert np.isfinite(net.y).all()


@pytest.mark.parametrize('control,rtol', [('regulator', 1e-7), ('bang_bang', 1e-7),
                                       ('regulator', 1e-4), ('bang_bang', 1e-5)])
def test_colored_table_template_preview_and_shutdown(control, rtol):
    cfg = config('pump_fed', control)
    cfg['simulation'] = {'fluid_solve_post_shutdown': False}
    cfg['advanced'] = {'fluid_network': {'max_step': .01, 'rtol': rtol, 'verify_jacobian': True}}
    with build(cfg, tabled=True) as prop:
        atm = SimpleNamespace(p=1e5)
        prop.update(None, atm, {})
        y = prop.network.y.copy()
        preview = prop.update(.2, atm, {}, commit=False)
        np.testing.assert_array_equal(prop.network.y, y)
        actual = prop.update(.2, atm, {})
        assert actual.propulsion.mode == preview.propulsion.mode == 'shutdown'
        assert actual.event_counts == preview.event_counts
        for key in actual.td_state:
            assert actual.td_state[key]['mass'] == pytest.approx(preview.td_state[key]['mass'])
