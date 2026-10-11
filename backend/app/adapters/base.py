from abc import ABC, abstractmethod
import numpy as np
from ..schema import ModelView

class Adapter(ABC):
    """One adapter per format family. Unsupported operations raise NotImplementedError."""
    format_ids: tuple = ()
    @abstractmethod
    def inspect(self) -> ModelView: ...
    def parameters(self) -> dict[str, np.ndarray]: raise NotImplementedError
    def predict(self, x: np.ndarray) -> np.ndarray: raise NotImplementedError
    def trace(self, x: np.ndarray) -> dict: raise NotImplementedError
    def with_edit(self, name: str, index: tuple, value: float) -> "Adapter": raise NotImplementedError
