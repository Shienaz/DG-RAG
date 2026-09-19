from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["GFMRetriever", "KGIndexer"]


def __getattr__(name: str) -> Any:
    if name == "GFMRetriever":
        return import_module("dgrag.retriever").GFMRetriever
    if name == "KGIndexer":
        return import_module("dgrag.kg_indexer").KGIndexer
    raise AttributeError(f"module 'dgrag' has no attribute {name!r}")
