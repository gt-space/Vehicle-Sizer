"""Stateless, node-local heat sources. All outputs are phase heat rates in W."""
from dataclasses import dataclass
from math import isfinite
from collections.abc import Mapping

from diagnostics.errors import TrialDomainError


def thermal_model(config):
    config = {} if config is None else config
    if not isinstance(config, Mapping):
        raise ValueError('thermal must be a mapping')
    model = config.get('model')
    if model in (None, 'None'):
        if set(config) - {'model'}:
            raise ValueError('Adiabatic thermal model does not accept correlation parameters')
        return 'None'
    if model not in HEAT_SOURCE_FACTORIES:
        raise ValueError(f'Unknown thermal model {model!r}')
    if model == 'Aeroheating' and set(config) - {'model', 'initial_wall_temperature'}:
        raise ValueError('Unknown Aeroheating parameters')
    return model


@dataclass(frozen=True)
class NoHeating:
    def evaluate(self, node_state):
        return {}


@dataclass(frozen=True)
class PrescribedHeatRates:
    rates: Mapping

    def evaluate(self, node_state):
        return dict(self.rates)


@dataclass(frozen=True)
class Aeroheating:
    def evaluate(self, node_state):
        raise ValueError('Aeroheating requires heat rates from the wall thermal solve')


@dataclass(frozen=True)
class DensityPowerLaw:
    area: float
    reference_area: float
    phase: str = 'gas'
    fluid: str | None = None
    reference_heat_rate: float = 15873.3
    reference_density: float = 209.559
    density_exponent: float = 1.2609

    def evaluate(self, node_state):
        fluids = [state for name, state in node_state.fluids.items()
                  if state.phase == self.phase and (self.fluid is None or name == self.fluid)]
        if len(fluids) != 1:
            raise ValueError('DensityPowerLaw requires exactly one matching fluid/phase')
        rho = float(fluids[0]['rho'])
        if not isfinite(rho) or rho <= 0:
            raise TrialDomainError('DensityPowerLaw requires finite positive trial density')
        try:
            value = self.area / self.reference_area * self.reference_heat_rate * (rho / self.reference_density)**self.density_exponent
        except OverflowError as error:
            raise TrialDomainError('DensityPowerLaw overflow') from error
        return {self.phase: value}


def _density_power_law(config, geometry):
    allowed = {'model', 'area', 'reference_area', 'phase', 'fluid',
               'reference_heat_rate', 'reference_density', 'density_exponent'}
    if set(config) - allowed:
        raise ValueError(f'Unknown DensityPowerLaw parameters: {set(config) - allowed}')
    area = config.get('area', getattr(geometry, 'internal_area', None))
    values = dict(area=area, reference_area=config.get('reference_area'),
                  reference_heat_rate=config.get('reference_heat_rate', 15873.3),
                  reference_density=config.get('reference_density', 209.559),
                  density_exponent=config.get('density_exponent', 1.2609))
    for key, value in values.items():
        if value is None or isinstance(value, bool) or not isfinite(float(value)):
            raise ValueError(f'DensityPowerLaw requires finite {key}')
        values[key] = float(value)
        if key in ('area', 'reference_area', 'reference_density') and values[key] <= 0:
            raise ValueError(f'DensityPowerLaw {key} must be positive')
    phase = config.get('phase', 'gas')
    if phase not in ('gas', 'liquid'):
        raise ValueError('DensityPowerLaw phase must be gas or liquid')
    return DensityPowerLaw(**values, phase=phase, fluid=config.get('fluid'))


HEAT_SOURCE_FACTORIES = {
    'None': lambda config, geometry: NoHeating(),
    'Aeroheating': lambda config, geometry: Aeroheating(),
    'DensityPowerLaw': _density_power_law,
}


def build_heat_source(config, geometry=None):
    """Resolve fixed geometry once; correlation evaluation reads only local state."""
    return HEAT_SOURCE_FACTORIES[thermal_model(config)](config or {}, geometry)


def evaluate_heat(source, state, time):
    """Adapt legacy time callbacks and prescribed mappings to the same contract."""
    if hasattr(source, 'evaluate'):
        rates = source.evaluate(state)
    else:
        rates = source(time) if callable(source) else source
        rates = PrescribedHeatRates(rates).evaluate(state)
    phases = {fluid.phase for fluid in state.fluids.values()}
    result = {}
    for phase, value in rates.items():
        value = float(value)
        if not isfinite(value):
            raise TrialDomainError('Heat source returned a nonfinite heat rate')
        if phase not in phases and value != 0:
            raise ValueError(f'Heat source targets absent phase {phase!r}')
        result[phase] = value
    return result
