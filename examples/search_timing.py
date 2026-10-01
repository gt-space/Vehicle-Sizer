"""Diagnostic-only nested wall/CPU timers; no model or solver behavior changes."""
from functools import wraps
import inspect
import json
from pathlib import Path
import time


class Timings:
    def __init__(self):
        self.records, self.stack = {}, []
        self.enabled = True

    def wrap(self, cls, method, label):
        descriptor = inspect.getattr_static(cls, method)
        original = getattr(cls, method)
        @wraps(original)
        def measured(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            key = label(*args, **kwargs) if callable(label) else label
            frame = [time.perf_counter(), time.process_time(), 0., 0.]
            self.stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                wall, cpu = time.perf_counter()-frame[0], time.process_time()-frame[1]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][2] += wall
                    self.stack[-1][3] += cpu
                record = self.records.setdefault(key, dict(calls=0, wall_s=0., cpu_s=0., self_wall_s=0., self_cpu_s=0.))
                record['calls'] += 1
                record['wall_s'] += wall
                record['cpu_s'] += cpu
                record['self_wall_s'] += max(0., wall-frame[2])
                record['self_cpu_s'] += max(0., cpu-frame[3])
        setattr(cls, method, staticmethod(measured) if isinstance(descriptor, staticmethod) else measured)

    def save(self, path):
        Path(path).write_text(json.dumps(self.records, indent=2))


def install():
    from Flight.Flight import FlightSim
    from Flight.environment import Environment
    from Vehicle.Vehicle import Vehicle
    from Thermals import ThermalNetwork
    from Fluids.FluidNetwork import FluidNetwork
    from Fluids.Sundials.ida_session import IdaSession
    from FluidTables.PropertyModels import TablePureFluidPropertySource, TableCombustionPropertySource
    from Fluids.FluidNode import (BoundaryComponent, JunctionComponent, VolumeComponent,
                                          PropellantTankComponent, CombustorComponent)
    from Fluids.FluidBranch import FluidBranch, RegulatorComponent, NozzleComponent
    timing = Timings()
    methods = {
        FlightSim: {'run': 'flight.total', '_attempt_segment': lambda self, kin, fluids, *args, **kwargs:
                    'flight.powered_attempt' if fluids.propulsion.mode == 'combusting' else 'flight.coast_attempt',
                    'step_coupled': 'flight.fluid_thermal_coupling', 'trial_aero': 'flight.aero',
                    'correct_kinematics': 'flight.kinematic_corrector', '_evaluate_loads': 'flight.loads'},
        Environment: {'__init__': 'setup.atmosphere', 'atmosphere': 'flight.atmosphere'},
        Vehicle: {'__init__': 'setup.vehicle', 'build': 'setup.vehicle_build',
                  'update_mass_distribution': 'flight.mass_distribution'},
        ThermalNetwork: {'trial': 'thermal.trial', 'commit': 'thermal.commit'},
        FluidNetwork: {'update': 'fluid.update', '_checkpoint': 'fluid.checkpoint', 'restore': 'fluid.rollback',
                       '_new_session': 'fluid.restart', '_apply_events': 'fluid.event_switch',
                       '_accept': 'fluid.acceptance', 'residual': 'fluid.residual',
                       'evaluate_trial': 'fluid.trial_states', 'root_values': 'fluid.root_functions'},
        IdaSession: {'advance': 'idas.advance', 'consistent': 'idas.consistent_initialization'},
        TablePureFluidPropertySource: {'__init__': 'setup.fluid_tables', 'state_pt': 'properties.state_pt',
            'derivatives_pt': 'properties.derivatives_pt', 'saturation_at_p': 'properties.saturation'},
        TableCombustionPropertySource: {'__init__': 'setup.cea_table', 'evaluate': 'properties.cea',
                                       'expansion_ratio': 'properties.expansion_ratio'},
    }
    for cls in (BoundaryComponent, JunctionComponent, VolumeComponent, PropellantTankComponent,
                CombustorComponent, FluidBranch, RegulatorComponent, NozzleComponent):
        methods[cls] = {method: f'component.{cls.__name__}.{method}'
                        for method in ('evaluate', 'residual') if method in cls.__dict__}
    for cls, entries in methods.items():
        for method, label in entries.items():
            timing.wrap(cls, method, label)
    return timing
