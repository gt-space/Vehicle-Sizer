from copy import deepcopy

import pytest

from Fluids.PropSystem import PropSystem
from Vehicle.Vehicle import Vehicle
import optimizer.core as opt
from Fluids.design import prepare_design_inputs
from Configs.loader import load_config
from FluidTables.PropertyModels import TableCombustionPropertySource


def test_real_table_interval_and_nozzle_response():
    source = TableCombustionPropertySource('FluidTables/sizer_lookups.h5', 1)
    low, high = source.design_exit_pressure_bounds((1.4e6, 3.5e6), (1.5, 3.))
    assert low == pytest.approx(40725.77119682981)
    assert high == pytest.approx(103421.3592)
    for pc in (1.4e6, 2.45e6, 3.5e6):
        for mr in (1.5, 2.25, 3.):
            large = source.expansion_ratio(pc, mr, 40726.)
            small = source.expansion_ratio(pc, mr, 103421.)
            assert 1 <= small < large <= 10


@pytest.mark.parametrize('bounds', [(100., 103421.), (40726., 104000.), (40726., 1.4e6)])
def test_unsupported_search_interval_rejected(bounds, tmp_path):
    settings = load_config('Configs/optimizer_pressure_fed.yaml')
    settings['bounds']['exit_pressure'] = bounds
    with pytest.raises(ValueError, match='Exit-pressure'):
        model = opt.DragModel()
        cfg, settings = opt.prepare(settings, model)
        pure, combustion = opt.property_sources(cfg)
        opt.Evaluator(cfg, settings, model, pure, combustion, tmp_path)


def test_exit_pressure_changes_sized_engine_without_mutating_base():
    cfg = load_config('Configs/flight_pressure_fed_regulator.yaml')
    original = deepcopy(cfg)
    pure, combustion = opt.property_sources(cfg)
    sizes = []
    for pressure in (40726., 103421.):
        candidate = deepcopy(cfg)
        candidate['engine']['exit_pressure'] = pressure
        template, states, wall_pressures = prepare_design_inputs(candidate)
        vehicle = Vehicle(candidate, pure, initial_states=states, wall_pressures=wall_pressures)
        with PropSystem(candidate, vehicle.tanks, pure, combustion, initial_states=states, template=template) as prop:
            sizes.append((prop.expansion_ratio, prop.exit_area, prop.throat_area))
    assert sizes[0][0] > sizes[1][0]
    assert sizes[0][1] > sizes[1][1]
    assert sizes[0][2] != sizes[1][2]
    assert cfg == original
