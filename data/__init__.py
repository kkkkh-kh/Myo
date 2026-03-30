"""Data package exports with lazy dataset import to avoid heavy dependencies at import time."""

from data.vocabulary import Vocabulary

__all__ = ["Vocabulary", "GlossChineseDataset"]


def __getattr__(name: str):
    if name == "GlossChineseDataset":
        from data.dataset import GlossChineseDataset

        return GlossChineseDataset
    raise AttributeError(f"module 'data' has no attribute {name!r}")
