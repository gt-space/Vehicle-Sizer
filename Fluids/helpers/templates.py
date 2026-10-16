"""Declarative network definitions shared by vehicle sizing and propulsion.

Only this loader reads template files. No feed-type dispatch or implicit wiring.
Paths are relative to the project root; absolute paths are also accepted.
"""
from copy import deepcopy
from pathlib import Path

import yaml


def read_template(cfg):
    """Read an unbound template so sweeps can still change config references."""
    selected = cfg["prop_system"].get("template")
    if selected is None:
        raise ValueError("prop_system.template must select a network template")
    if isinstance(selected, str):
        path = Path(selected)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        with path.open() as stream:
            template = yaml.safe_load(stream)
    else:
        template = deepcopy(selected)

    return template


def bind_template(cfg):
    """Bind config references and validate wiring, without evaluating pressures."""
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

    template = bind(read_template(cfg))
    if not isinstance(template, dict) or set(template) != {"circuits", "nodes", "branches"}:
        raise ValueError("Template requires exactly circuits, nodes and branches mappings")
    if any(not isinstance(value, dict) for value in template.values()):
        raise ValueError("Template circuits, nodes and branches must be mappings")
    nodes, branches = template["nodes"], template["branches"]
    for key, definition in (*nodes.items(), *branches.items()):
        if "component" not in definition:
            raise ValueError(f"Template component {key!r} requires a component type")
    for key, branch in branches.items():
        parameters = branch.pop("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"Branch {key!r} parameters must be a mapping")
        if set(parameters) & {"component", "from", "to", "from_port", "to_port", "circuit", "pump_id"}:
            raise ValueError(f"Branch {key!r} parameters cannot override template wiring or identity")
        if set(parameters) & branch.keys():
            raise ValueError(f"Branch {key!r} repeats parameters in its template")
        branch.update(parameters)
        if branch.get("from") not in nodes or branch.get("to") not in nodes:
            raise ValueError(f"Branch {key!r} references an unknown node")
        if branch.get("circuit") not in template["circuits"]:
            raise ValueError(f"Branch {key!r} references an unknown circuit")

    return template
