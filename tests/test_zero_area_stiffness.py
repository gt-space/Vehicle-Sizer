from types import SimpleNamespace
import numpy as np
from Flight.loads import Loads
from Thermals.ThermalNode import WetNodeModel


def test_zero_ei_preserves_loads_without_dividing():
    vehicle = SimpleNamespace(EI=np.array([1., 0., 2.]), station=np.array([.5, 1.5, 2.5]),
                              cell_edges=np.arange(4.), length=3., cg=1.5)
    loads = Loads(vehicle, SimpleNamespace(reference_area=1.))
    with np.errstate(all='raise'):
        shear, bending, slope, displacement = loads.beam_deflection(np.array([1., -2., 1.]))
    np.testing.assert_allclose(shear, [1, -1, 0])
    np.testing.assert_allclose(bending, [1, 0, 0])
    assert np.isnan(slope).all() and np.isnan(displacement).all()


def test_zero_contact_area_has_zero_conductance(monkeypatch):
    import Thermals.ThermalNode as module
    model = WetNodeModel()
    phase = {'T': 290., 'rho': 1000., 'mu': .001, 'k': .1, 'cp': 1000., 'beta': .001}
    monkeypatch.setattr(model, '_phases', lambda *args: ('fuel', {'liquid': phase}, np.array(['liquid']*2)))
    monkeypatch.setattr(module, 'natural_convection_htc', lambda *args: (np.array([10.,10.]), np.ones(2)))
    node = SimpleNamespace(n=2, id='fuel', cell_edges=np.arange(3.),
                           internal_area=np.array([0., 1.]), thickness=.003, conductivity=100.)
    with np.errstate(all='raise'):
        result = model.fluid_boundary(node, None, np.array([300.,300.]), 9.81)
    assert result[0][0] == 0
    assert result[0][1] > 0
