"""Exercise the installed material API used by the current flight configs."""
import numpy as np
import pytest

from Configs.loader import load_config
from Vehicle.Material import MaterialProperties, mp
from Vehicle.sections.AviBay import AviBay


@pytest.mark.parametrize('architecture', ['pump_fed', 'pressure_fed'])
def test_material_names_and_strong_composite_bulkhead(architecture):
    cfg = load_config(f'Configs/flight_{architecture}_regulator.yaml')
    for material in [tank['material'] for tank in cfg['tanks'].values()]:
        properties = MaterialProperties.from_name(material)
        assert properties.density > 0
        assert properties.require('yield_strength') > 0
    bay = AviBay(cfg)
    material = mp.get_material(cfg['avi_bay']['bulkhead_material'])
    pressure_load = cfg['avi_bay']['avi_mass'] * 9.81 * 10
    radius = cfg['vehicle']['OMLD']/2 - cfg['avi_bay']['clamshell_thickness']
    nu = material.get('poisson_ratio', T=350.)
    strength = material.get('yield_strength', T=350.)
    thickness = bay._get_bulkhead_thickness(pressure_load, radius, nu, strength) / 1.5
    stress = pressure_load/thickness**2 * (1 + nu) * (.485*np.log(radius/thickness) + .52)
    assert stress == pytest.approx(strength, rel=1e-7)
    if architecture == 'pressure_fed':
        assert thickness < .001
    assert bay._get_bulkhead_mass() > 0


@pytest.mark.parametrize('architecture', ['pump_fed', 'pressure_fed'])
@pytest.mark.parametrize('control', ['regulator', 'bang_bang'])
def test_configurable_shell_and_hardware_masses(architecture, control):
    from Vehicle.Engine import Engine
    from Vehicle.sections.Nosecone import Nosecone
    from Vehicle.sections.FinCan import FinCan

    cfg = load_config(f'Configs/flight_{architecture}_{control}.yaml')
    nose = Nosecone(cfg)
    nose.get_mass()
    shell = nose.shell_mass.sum()
    cfg['nosecone']['wall_thickness'] *= 2
    thicker = Nosecone(cfg)
    thicker.get_mass()
    assert thicker.mass.sum() - nose.mass.sum() == pytest.approx(shell)

    fin = FinCan(cfg, Engine(cfg['engine']['mass'], cfg['engine']['length'], .01))
    fin.get_mass()
    baseline = fin.mass.sum()
    cfg['fin_can']['hardware_mass'] += 3
    fin.get_mass()
    assert fin.mass.sum() - baseline == pytest.approx(3)
