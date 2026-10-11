"""Model adapters registry."""

from app.adapters.base import Adapter
from app.adapters.tree import TreeAdapter
from app.adapters.keras import KerasAdapter

__all__ = ["Adapter", "TreeAdapter", "KerasAdapter"]