from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Capability(BaseModel):
    status: str
    why: str


class ModelNode(BaseModel):
    id: str
    label: str
    type: str
    attrs: Dict[str, Any] = {}


class ModelEdge(BaseModel):
    source: str
    target: str


class ModelView(BaseModel):
    kind: str
    name: str
    nodes: List[ModelNode]
    edges: List[ModelEdge]
    meta: Dict[str, Any] = {}
    capabilities: Dict[str, Capability] = {}
