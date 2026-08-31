from scipy.optimize import root
from .record import initialize_data, record


def solve(vehicle, dt: float, t_final: float) -> dict:
    """Uses root to solve each timestep. Also stores the data"""
    network = vehicle.network
    time = 0.0

    network.evaluate()
    states = network.collect_states()
    stored_states = network.collect_stored_states()
    data = initialize_data(network)
    record(data, network, time)

    while time < t_final:
        previous = [state.value for state in stored_states]
        x0 = [state.value for state in states]

        solution = root(
            residual,
            x0,
            args=(vehicle, dt, states, stored_states, previous),
        )
        if not solution.success:
            raise RuntimeError(
                f"Vehicle simulation failed at t={time}: {solution.message}"
            )

        for state, value in zip(states, solution.x):
            state.value = float(value)

        network.evaluate()
        time += dt
        record(data, network, time)

    return data


def residual(x, vehicle, dt, states, stored_states, previous):
    """Makes the backward Euler residual function for each timestep"""
    network = vehicle.network
    for state, value in zip(states, x):
        state.value = value

    vehicle.evaluate()

    derivatives = network.collect_derivatives()
    balances = network.collect_balances()
    dynamics = [state.value - old - dt * derivative for state, old, derivative in zip(stored_states, previous, derivatives)]
    return [float(value) for value in dynamics + balances]
