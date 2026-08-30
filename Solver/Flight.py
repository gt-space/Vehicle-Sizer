from __future__ import annotations
from numbers import Real
from scipy.optimize import root
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Layout import Vehicle


class Flight:

    def __init__(self, vehicle: Vehicle, dt: float, t_final: float):

        self.vehicle = vehicle
        self.dt = dt
        self.t_final = t_final

        self.network = self.vehicle.network
        self.time = 0.0
        self.data = {}


    def simulate(self):
        # Evaluate initial conditions.
        self.network.evaluate()
        states = self.network.collect_states()
        stored_states = self.network.collect_stored_states()
        # Initialize history.
        self._initialize_data()
        self._record()
        while self.time < self.t_final:
            # Snapshot conserved quantities at time n.
            previous = [state.value for state in stored_states]
            # Use the current solution as the initial guess for n + 1.
            x0 = [state.value for state in states]

            solution = root(self._residual, x0, args=(states, stored_states, previous))

            if not solution.success:
                raise RuntimeError(
                    f"Flight simulation failed at t={self.time}: "
                    f"{solution.message}"
                )
            # Apply the converged solution.
            for state, value in zip(states, solution.x):
                state.value = float(value)

            self.network.evaluate()
            self.time += self.dt
            self._record()
        return self.data



    def _residual(self, x, states, stored_states, previous):
        # Apply scipy's current guess.
        for state, value in zip(states, x):
            state.value = value
        # Re-evaluate all component physics.
        self.network.evaluate()
        derivatives = self.network.collect_derivatives()
        balances = self.network.collect_balances()

        dynamic_residuals = [state.value - old - self.dt * derivative for state, old, derivative in zip(stored_states, previous, derivatives)]

        return [float(value) for value in dynamic_residuals + balances]



    def _initialize_data(self):
        self.data = {"time": []}
        for component in self.network.components:
            self.data[component.name] = {}
            for name, value in vars(component).items():
                if name in ("name", "network"):
                    continue
                if hasattr(value, "value"):
                    value = value.value
                if isinstance(value, Real):
                    self.data[component.name][name] = []



    def _record(self):
        self.data["time"].append(self.time)
        for component in self.network.components:
            for name in self.data[component.name]:
                value = getattr(component, name)
                if hasattr(value, "value"):
                    value = value.value
                self.data[component.name][name].append(value)
