from importlib import import_module
from typing import Any

__all__ = ["BaseKGConstructor", "KGConstructor", "BaseQAConstructor", "QAConstructor"]


def __getattr__(name: str) -> Any:
    if name in {"BaseKGConstructor", "KGConstructor"}:
        module = import_module("dgrag.kg_construction.kg_constructor")
        return getattr(module, name)
    if name in {"BaseQAConstructor", "QAConstructor"}:
        module = import_module("dgrag.kg_construction.qa_constructor")
        return getattr(module, name)
    raise AttributeError(f"module 'dgrag.kg_construction' has no attribute {name!r}")
