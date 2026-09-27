from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from Configs.loader import load_config
from simulation import property_sources
from Vehicle.Engine import Engine
from Vehicle.Material import MaterialProperties
from Vehicle.Vehicle import Vehicle
from Vehicle.sections.InterTank import InterTank


@pytest.mark.parametrize('override', [False, True])
def test_per_intertank_mass_changes_only_its_section_and_total_dry_mass(override):
    cfg = load_config('Configs/flight_pump_fed_regulator.yaml')
    for section in cfg['vehicle']['sections']:
        if section['type'] == 'inter_tank':
            if not override:
                section.pop('mass')
            # With an override, even these masses must not be added again.
            section.update(feed_system_mass=2.0, avi_mass=1.0)
    pure, _ = property_sources(cfg)

    def build(candidate):
        vehicle = Vehicle(candidate, pure)
        vehicle.build(Engine(float(candidate['engine']['mass']),
                             float(candidate['engine']['length']), .02))
        return vehicle

    baseline = build(cfg)
    indices = [i for i, s in enumerate(cfg['vehicle']['sections']) if s['type'] == 'inter_tank']
    for index in indices:
        section = baseline.sections[index]
        assigned = cfg['vehicle']['sections'][index]
        if override:
            expected = assigned['mass']
        else:
            rho = MaterialProperties.from_name(cfg['inter_tank']['stringer_material']).density
            expected = (np.sum(section.shell_mass) + section.stringer_thickness**2
                        * section.length * section.stringer_count * rho + 3.0)
        assert np.sum(section.mass) == pytest.approx(expected)
        assert section.stringer_thickness > 0
        assert np.all(section.EI > 0)
    original = deepcopy(cfg)
    for index in indices:
        candidate = deepcopy(cfg)
        if override:
            candidate['vehicle']['sections'][index]['mass'] += 2.5
        else:
            candidate['vehicle']['sections'][index]['feed_system_mass'] += 2.0
            candidate['vehicle']['sections'][index]['avi_mass'] += 0.5
        changed = build(candidate)
        assert np.sum(changed.dry_mass) - np.sum(baseline.dry_mass) == pytest.approx(2.5)
        for i, (old, new) in enumerate(zip(baseline.sections, changed.sections)):
            assert np.sum(new.mass) - np.sum(old.mass) == pytest.approx(2.5 if i == index else 0)
    assert cfg == original


@pytest.mark.parametrize('field', ['feed_system_mass', 'avi_mass'])
@pytest.mark.parametrize('value', [-1, float('nan'), float('inf')])
def test_invalid_section_hardware_mass_is_rejected(field, value):
    masses = dict(feed_system_mass=0, avi_mass=0)
    masses[field] = value
    with pytest.raises(ValueError, match='finite and nonnegative'):
        InterTank({'vehicle': {'dx': .01}}, .4, 3.6e-5, **masses)


def test_all_active_flight_configs_specify_hardware_per_section():
    for path in Path('Configs').glob('flight_*.yaml'):
        cfg = load_config(path)
        assert 'feed_system_mass' not in cfg['inter_tank']
        assert 'avi_mass' not in cfg['inter_tank']
        for section in cfg['vehicle']['sections']:
            if section['type'] == 'inter_tank':
                if 'mass' in section:
                    assert section['mass'] > 0
                else:
                    assert section.get('feed_system_mass', 0) >= 0
                    assert section.get('avi_mass', 0) >= 0


@pytest.mark.parametrize('mass', [0, -1, float('nan'), float('inf')])
def test_invalid_total_override_is_rejected(mass):
    with pytest.raises(ValueError, match='override must be finite and positive'):
        InterTank({'vehicle': {'dx': .01}}, .4, 3.6e-5, mass=mass)
