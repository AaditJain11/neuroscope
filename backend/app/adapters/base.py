"""Base adapter class defining canonical interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
from app.schemas import ModelView


class Adapter(ABC):
    """Canonical abstract adapter for all ML framework formats."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    @abstractmethod
    def inspect(self) -> ModelView:
        """Parse topology and attributes into canonical ModelView schema."""
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Execute forward pass inference."""
        pass

    @abstractmethod
    def get_activations(self, input_data: Any) -> Dict[str, Any]:
        """Trace layer activations."""
        pass