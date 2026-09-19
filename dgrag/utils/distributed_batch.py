from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import distributed as dist


SINGLE_BATCH_EDGE_PARALLEL = "single_batch_edge_parallel"


def use_single_batch_edge_parallel(cfg: Any) -> bool:
    train_cfg = getattr(cfg, "train", cfg)
    if hasattr(train_cfg, "__contains__") and "distributed_batch_mode" not in train_cfg:
        return False
    return (
        getattr(train_cfg, "distributed_batch_mode", "ddp")
        == SINGLE_BATCH_EDGE_PARALLEL
    )


def broadcast_object(value: Any, src: int = 0) -> Any:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    objects = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def next_shared_batch(iterator: Iterator | None, src: int = 0) -> Any | None:
    value = None
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == src:
        try:
            value = next(iterator) if iterator is not None else None
        except StopIteration:
            value = None
    return broadcast_object(value, src=src)


def require_distributed_single_batch_mode(cfg: Any) -> None:
    if not use_single_batch_edge_parallel(cfg):
        return
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        raise ValueError(
            "single_batch_edge_parallel requires torchrun with world_size > 1"
        )


def validate_single_batch_edge_parallel_config(cfg: Any) -> None:
    """Validate options that are unsafe when all ranks share one batch."""
    if not use_single_batch_edge_parallel(cfg):
        return

    entity_model_cfg = getattr(getattr(cfg, "model", None), "entity_model", None)
    if not bool(getattr(entity_model_cfg, "use_edge_parallel", False)):
        raise ValueError(
            "single_batch_edge_parallel requires model.entity_model.use_edge_parallel=true; "
            "otherwise every rank runs the same full graph without edge sharding."
        )

    model_cfg = getattr(cfg, "model", None)
    if bool(getattr(model_cfg, "use_question_entity_contrastive", False)):
        qec_mode = getattr(model_cfg, "question_entity_contrastive_mode", "batch")
        if qec_mode != "hard":
            raise ValueError(
                "single_batch_edge_parallel shares the same QA sample across ranks, "
                "so batch/hybrid question-entity contrastive would gather duplicate "
                "positives as negatives. Use model.question_entity_contrastive_mode=hard "
                "or disable QEC."
            )


def expand_single_batch_for_bidirectional_negatives(batch: torch.Tensor) -> torch.Tensor:
    """Duplicate one KGC triple so strict negative sampling covers both directions."""
    if batch.dim() == 2 and batch.size(0) == 1 and batch.size(-1) == 3:
        return batch.expand(2, -1).clone()
    return batch
