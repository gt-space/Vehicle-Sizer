"""Power-limited battery mass sizing; all inputs use SI units."""
from math import ceil, isfinite
from collections.abc import Mapping


def size_battery(definition, power_draw_w):
    """Size integer series/parallel cell counts from electrical pump demand.

    This model does not check stored energy or packaging. Pack voltage is the
    prescribed operating voltage, even when series-cell rounding raises nominal
    voltage. Margins cover unmodeled pack hardware without sizing its geometry.
    """
    if not isinstance(definition, Mapping) or not isinstance(definition.get('cell'), Mapping):
        raise ValueError('prop_system.battery and its cell must be mappings')

    def number(value, name, minimum=0., inclusive=False):
        if isinstance(value, bool):
            raise ValueError(f'Battery {name} must be numeric')
        value = float(value)
        if not isfinite(value) or (value < minimum if inclusive else value <= minimum):
            raise ValueError(f'Invalid battery {name}: {value}')
        return value

    cell = definition['cell']
    power = number(power_draw_w, 'power_draw_w')
    voltage = number(definition['pack_voltage'], 'pack_voltage')
    cell_voltage = number(cell['nominal_voltage'], 'cell.nominal_voltage')
    cell_current = number(cell['max_cont_discharge_current'], 'cell.max_cont_discharge_current')
    cell_mass = number(cell['mass'], 'cell.mass')
    fos = number(definition.get('current_fos', 1.7), 'current_fos', 1., True)
    margin = number(definition.get('mass_margin', 1.3), 'mass_margin', 1., True)
    additional = number(definition.get('additional_mass', 0.), 'additional_mass', 0., True)
    series = ceil(voltage / cell_voltage)
    current = power / voltage
    allowable = cell_current / fos
    parallel = ceil(current / allowable)
    mass = series * parallel * cell_mass
    return dict(power_draw_w=power, pack_voltage=voltage,
                nominal_pack_voltage=series * cell_voltage, pack_current_a=current,
                allowable_cell_current_a=allowable, cells_in_series=series,
                cells_in_parallel=parallel, cell_count=series * parallel,
                cell_mass_kg=mass, mass_margin=margin, additional_mass_kg=additional,
                calculated_mass_kg=mass * margin + additional)
