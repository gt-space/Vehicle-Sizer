"""Integration checks against the expanded HDF5 deck, not synthetic CA shapes."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import trapezoid

from AeroTables import DragModel
from Flight.flight_forces import Aero
from Flight.loads import Loads
from test_aero import CANDIDATE


@pytest.fixture(scope="module")
def model():
    return DragModel()


@pytest.mark.parametrize("length,exit", [(309.26, 6.79), (433.20, 7.75)])
@pytest.mark.parametrize("drag_multiplier", [1.0, 1.3, 0.0])
def test_unclipped_geometry_and_force_closure_on_actual_deck(model, length, exit, drag_multiplier):
    candidate = {**CANDIDATE, "length": length, "exit": exit}
    aero = Aero(dict(aoa_schedule=[[0, 0], [10, 15]], fins_on_boattail=True,
                     drag_multiplier=drag_multiplier), candidate, model)
    assert aero.candidate == candidate
    for cells in (17, 137):
        # Nonuniform grid deliberately not aligned with the 800-point aero grid.
        edges = np.linspace(0, 1, cells+1)**1.2 * length*.0254
        vehicle = SimpleNamespace(cell_edges=edges, station=(edges[:-1]+edges[1:])/2,
                                  engine_start_station=edges[-4], mass=np.ones(cells),
                                  total_mass=float(cells))
        loads = Loads(vehicle, aero)
        for mach in (.0, .97, 2., 10.):
            for alpha in (0., 7., -7., 15.):
                for on in (False, True):
                    angle = np.deg2rad(alpha)
                    distribution = aero.axial_distribution(mach, angle, on)
                    ca = aero.coefficients(mach, angle, on)[1]
                    assert trapezoid(distribution["dca_dx"], distribution["x"]) == pytest.approx(ca, abs=1e-10)
                    forces = loads.get_axial_forces(3000, mach, angle, on)
                    assert forces.sum() == pytest.approx(3000*aero.reference_area*ca, abs=1e-8)
                    assert abs(loads.get_axial_load(forces, 10000)[-1]) < 1e-8


def test_adapter_units_sign_and_base_credit(model):
    aero = Aero(dict(aoa_schedule=[[0, 0], [10, 15]]), CANDIDATE, model)
    raw = model.ca_distribution(CANDIDATE, 2., 7.)
    off = aero.axial_distribution(2., np.deg2rad(-7.), False)
    on = aero.axial_distribution(2., np.deg2rad(7.), True)
    np.testing.assert_allclose(off["x"], raw["x"]*.0254)
    np.testing.assert_allclose(off["diameter"], raw["diameter"]*.0254)
    # Match fin placement in the raw reader before comparing densities.
    raw = model.ca_distribution(CANDIDATE, 2., 7., fins_on_boattail=True)
    np.testing.assert_allclose(off["dca_dx"], raw["dca_dx"]/.0254)
    for key in off["parts"]:
        if key != "base":
            np.testing.assert_allclose(off["parts"][key], on["parts"][key])
    assert off["point_loads"]["base"][0] == pytest.approx(CANDIDATE["length"]*.0254)
    assert off["point_loads"]["base"][1] - on["point_loads"]["base"][1] == pytest.approx(off["ca"]-on["ca"])


@pytest.mark.parametrize("change", [{"exit": 5.8}, {"length": 501}, {"length": 275}])
def test_actual_deck_rejects_out_of_bounds(model, change):
    with pytest.raises(ValueError):
        Aero(dict(aoa_schedule=[[0, 0], [10, 0]]), {**CANDIDATE, **change}, model)
