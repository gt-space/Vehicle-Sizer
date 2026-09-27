import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

sys.modules.pop("Thermals.heating", None)
heating = importlib.import_module("Thermals.heating")


def test_weak_shock_angle_matches_nasa_wedge_case():
    beta = heating.get_oblique_beta(2.5, np.deg2rad(15.0), 1.4)
    assert np.rad2deg(beta) == pytest.approx(36.9449, abs=1.0e-4)


def test_oblique_shock_rejects_invalid_regimes():
    with pytest.raises(ValueError, match="Mach > 1"):
        heating.get_oblique_beta(0.8, 0.0, 1.4)
    with pytest.raises(ValueError, match="No attached"):
        heating.get_oblique_beta(2.0, np.deg2rad(30.0), 1.4)


def test_reference_properties_use_reference_temperature():
    wall_temperature = np.array([300.0, 350.0])
    with (
        patch.object(heating, "get_cp", return_value=1000.0) as get_cp,
        patch.object(heating, "get_k", return_value=0.03) as get_k,
    ):
        reference_temperature, *_ = heating.get_ref_props(
            500.0, wall_temperature, 101325.0, 2.0, 1.4, 0.72
        )

    np.testing.assert_allclose(get_cp.call_args.args[0], reference_temperature)
    np.testing.assert_allclose(get_k.call_args.args[0], reference_temperature)


def test_knudsen_check_warns_outside_continuum_regime():
    atmosphere = SimpleNamespace(T=288.15, p=101325.0, mu=1.789e-5)
    assert heating.get_Kn(atmosphere, 1.0) == pytest.approx(6.3e-8, rel=0.05)
    with pytest.warns(RuntimeWarning, match="Kn >= 0.01"):
        heating.check_continuum(atmosphere, 1.0e-7)


def test_subsonic_heating_uses_freestream_without_a_shock():
    atmosphere = SimpleNamespace(
        T=288.15,
        p=101325.0,
        rho=1.225,
        mu=1.789e-5,
        a=340.0,
        Ma=0.5,
    )
    heat_flux = heating.get_body_heating(
        np.array([0.5, 1.0]),
        np.array([280.0, 280.0]),
        atmosphere,
        theta=np.deg2rad(5.0),
    )

    assert np.all(np.isfinite(heat_flux))
    assert np.all(heat_flux > 0.0)
