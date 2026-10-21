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
    if model not in ('Aeroheating', 'Prescribed'):
        raise ValueError(f'Unknown thermal model {model!r}')
    if model == 'Aeroheating' and set(config) - {'model', 'initial_wall_temperature'}:
        raise ValueError('Unknown Aeroheating parameters')
    if model == 'Prescribed':
        if set(config) - {'model', 'heat_rate'}:
            raise ValueError('Unknown Prescribed parameters')
        rates = config.get('heat_rate')
        if not isinstance(rates, Mapping) and not callable(rates):
            raise ValueError('Prescribed heat_rate must be a phase-to-watts mapping or time callback')
        if isinstance(rates, Mapping):
            if set(rates) - {'gas', 'liquid'}:
                raise ValueError('Prescribed heat_rate phases must be gas or liquid')
            if any(not isfinite(float(value)) for value in rates.values()):
                raise ValueError('Prescribed heat_rate must be finite')
    return model


@dataclass(frozen=True)
class NoHeating:
    def evaluate(self, node_state):
        return {}


@dataclass(frozen=True)
class Aeroheating:
    def evaluate(self, node_state):
        raise ValueError('Aeroheating requires heat rates from the wall thermal solve')


def build_heat_source(config):
    """Select wall heating, prescribed phase rates, or zero heating."""
    model = thermal_model(config)
    if model == 'Prescribed':
        rates = config['heat_rate']
        return dict(rates) if isinstance(rates, Mapping) else rates
    return Aeroheating() if model == 'Aeroheating' else NoHeating()


def evaluate_heat(source, state, time):
    """Validate phase heat rates from a state model, time callback or mapping."""
    if hasattr(source, 'evaluate'):
        rates = source.evaluate(state)
    else:
        rates = source(time) if callable(source) else source
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
