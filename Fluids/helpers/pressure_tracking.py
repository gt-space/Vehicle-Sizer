"""Accepted powered-interval pressure errors; owned by the network checkpoint."""
from math import isfinite


def absolute_linear_integral(a, b, dt):
    """Exact integral of the absolute value of a linearly interpolated error."""
    if a * b < 0:
        return dt * (a*a + b*b) / (2 * (abs(a) + abs(b)))
    return dt * (abs(a) + abs(b)) / 2


class PressureTracking:
    def __init__(self, definitions):
        self.targets = {key: (node.get('tank_id', key), float(node['pressure_tracking_target']))
                        for key, node in definitions.items() if 'pressure_tracking_target' in node}
        if any(not isfinite(p) or p <= 0 for _, p in self.targets.values()):
            raise ValueError('Pressure tracking targets must be finite and positive')
        self.records = {}
        self.previous = None
        self.ended = False

    def accept(self, time, nodes, powered):
        if self.ended:
            return
        if not powered:
            self.ended = self.previous is not None
            return
        errors = {tank: float(nodes[key]['P']) - target for key, (tank, target) in self.targets.items()}
        if not isfinite(time) or any(not isfinite(e) for e in errors.values()):
            raise ValueError('Nonfinite pressure-tracking sample')
        dt = 0. if self.previous is None else time - self.previous[0]
        if dt < 0:
            raise ValueError('Pressure tracking moved backward without restoring checkpoint')
        for key, (tank, target) in self.targets.items():
            record = self.records.setdefault(tank, dict(target_pressure_pa=target,
                integral_pa_s=0., duration_s=0., peak_error_pa=0.))
            error = errors[tank]
            if dt:
                record['integral_pa_s'] += absolute_linear_integral(self.previous[1][tank], error, dt)
                record['duration_s'] += dt
            record['peak_error_pa'] = max(record['peak_error_pa'], abs(error))
        self.previous = (time, errors)
