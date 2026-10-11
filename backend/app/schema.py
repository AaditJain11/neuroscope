"""Common internal representation. Adapters fill this; the frontend picks a view by `kind`."""
from dataclasses import dataclass, field, asdict
from typing import Any

SUPPORTED, PARTIAL, UNSUPPORTED = "supported", "partial", "unsupported"
CAPABILITIES = ["architecture", "parameters", "graph", "inference",
                "intermediates", "param_editing", "retraining", "export"]

@dataclass
class Capability:
    status: str
    why: str = ""

@dataclass
class Node:
    id: str
    label: str
    type: str
    attrs: dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    source: str
    target: str

@dataclass
class ModelView:
    kind: str                       # "tree" | "graph" | "dense" ...
    name: str
    nodes: list[Node]
    edges: list[Edge]
    meta: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Capability] = field(default_factory=dict)
    def to_dict(self): return asdict(self)
