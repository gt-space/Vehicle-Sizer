def initialize_data(network) -> dict:
    """Makes an empty dict to store data"""
    data = {"time": []}
    for component in network.components:
        data[component.name] = {}
        for name, value in vars(component).items():
            if name in ("name", "network"):
                continue
            if hasattr(value, "value"):
                value = value.value
            if isinstance(value, (int, float)):
                data[component.name][name] = []
    return data


def record(data: dict, network, time: float) -> None:
    """Stores transient data to dict"""
    data["time"].append(time)
    for component in network.components:
        for name in data[component.name]:
            value = getattr(component, name)
            if hasattr(value, "value"):
                value = value.value
            data[component.name][name].append(value)
