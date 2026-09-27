"""Declarative network definitions shared by vehicle sizing and propulsion.

Only this loader reads template files. No feed-type dispatch or implicit wiring.
Paths are relative to the project root; absolute paths are also accepted.
"""
from copy import deepcopy
from math import isfinite
from pathlib import Path

import yaml


def load_template(cfg, *, validate_pressures=True):
    selected = cfg["prop_system"].get("template")
    if selected is None:
        raise ValueError("prop_system.template must select a network template")
    if isinstance(selected, str):
        path = Path(selected)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        with path.open() as stream:
            template = yaml.safe_load(stream)
    else:
        template = deepcopy(selected)

    def bind(value):
        if isinstance(value, dict):
            if set(value) == {"config"}:
                source = cfg
                try:
                    for part in value["config"].split("."):
                        source = source[part]
                except (KeyError, TypeError) as error:
                    raise ValueError(f"Unknown template config reference: {value['config']}") from error
                return deepcopy(source)
            return {key: bind(item) for key, item in value.items()}
        if isinstance(value, list):
            return [bind(item) for item in value]
        return value

    template = bind(template)
    if not isinstance(template, dict) or set(template) != {"circuits", "nodes", "branches"}:
        raise ValueError("Template requires exactly circuits, nodes and branches mappings")
    if any(not isinstance(value, dict) for value in template.values()):
        raise ValueError("Template circuits, nodes and branches must be mappings")
    nodes, branches = template["nodes"], template["branches"]
    for key, definition in (*nodes.items(), *branches.items()):
        kinds = [name for name in ("component", "model") if name in definition]
        if len(kinds) != 1:
            raise ValueError(f"Template component {key!r} needs exactly one component/model type")
    for key, branch in branches.items():
        parameters = branch.pop("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"Branch {key!r} parameters must be a mapping")
        if set(parameters) & {"component", "model", "from", "to", "from_port", "to_port", "circuit", "pump_id"}:
            raise ValueError(f"Branch {key!r} parameters cannot override template wiring or identity")
        if set(parameters) & branch.keys():
            raise ValueError(f"Branch {key!r} repeats parameters in its template")
        branch.update(parameters)
        if branch.get("from") not in nodes or branch.get("to") not in nodes:
            raise ValueError(f"Branch {key!r} references an unknown node")
        if branch.get("circuit") not in template["circuits"]:
            raise ValueError(f"Branch {key!r} references an unknown circuit")

    visiting = set()

    def pressure(key):
        if key in visiting:
            raise ValueError(f"Cyclic design-pressure references at {key!r}")
        if key not in nodes or "P0" not in nodes[key]:
            raise ValueError(f"Missing design pressure for node {key!r}")
        value = nodes[key]["P0"]
        if isinstance(value, dict):
            if "node" not in value or set(value) - {"node", "relative_rise", "rise", "drop"}:
                raise ValueError(f"Invalid design-pressure relation for {key!r}")
            visiting.add(key)
            value = (pressure(value["node"]) * (1 + float(value.get("relative_rise", 0)))
                     + float(value.get("rise", 0)) - float(value.get("drop", 0)))
            visiting.remove(key)
        value = float(value)
        if not isfinite(value) or (validate_pressures and value <= 0):
            raise ValueError(f"Node {key!r} design pressure must be finite and positive")
        nodes[key]["P0"] = value
        return value

    for key, node in nodes.items():
        if "P0" in node:
            pressure(key)
        state = node.get('state0', {})
        if isinstance(state.get('P'), dict) and set(state['P']) == {'node'}:
            state['P'] = pressure(state['P']['node'])
    return template
