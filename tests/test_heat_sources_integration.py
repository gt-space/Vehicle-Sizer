"""Energy balance, rollback and Jacobian coverage for prescribed heating."""
from copy import deepcopy

import numpy as np
import pytest

from Fluids.FluidNetwork import FluidNetwork
from test_fluid_network import gas_tank, Properties


def network(model):
    definition = gas_tank()
    definition['thermal'] = model
    return FluidNetwork({'tank': definition}, {}, fluid_properties=Properties(),
                        tolerances={'verify_jacobian': True})


def test_closed_volume_energy_and_preview_rollback():
    config = {'model': 'Prescribed', 'heat_rate': {'gas': 100.}}
    with network(config) as net:
        first = net.initialize()
        qdot = 100.
        initial_energy = first['node']['tank']['U']
        original = net.y.copy()
        preview = net.update(.2, commit=False)
        np.testing.assert_array_equal(net.y, original)
        assert net.time == 0.
        committed = net.update(.2)
        assert committed['node']['tank']['U'] == pytest.approx(initial_energy + .2 * qdot, rel=1e-6)
        assert committed['node']['tank']['U'] == pytest.approx(preview['node']['tank']['U'])
        assert committed['node']['tank']['heat_rate']['gas'] == pytest.approx(qdot)


def test_none_is_adiabatic_and_explicit_override_conflicts_with_prescribed_source():
    with network({'model': None}) as net:
        initial = net.initialize()['node']['tank']['U']
        assert net.update(.2)['node']['tank']['U'] == pytest.approx(initial)
    with network({'model': 'Prescribed', 'heat_rate': {'gas': 100.}}) as net:
        with pytest.raises(ValueError, match='Conflicting'):
            net.initialize(heat_rate={'tank': {'gas': 1.}})


def test_mixed_models_in_one_network():
    nodes = {key: gas_tank() for key in ('aero', 'prescribed', 'adiabatic')}
    nodes['aero']['thermal'] = {'model': 'Aeroheating'}
    nodes['prescribed']['thermal'] = {'model': 'Prescribed', 'heat_rate': {'gas': 100.}}
    with FluidNetwork(nodes, {}, fluid_properties=Properties()) as net:
        initial = deepcopy(net.initialize(heat_rate={'aero': {'gas': 50.}})['node'])
        result = net.update(.1)['node']
        assert result['aero']['U'] - initial['aero']['U'] == pytest.approx(5.)
        assert result['prescribed']['U'] - initial['prescribed']['U'] == pytest.approx(10.)
        assert result['adiabatic']['U'] == pytest.approx(initial['adiabatic']['U'])
