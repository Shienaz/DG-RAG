import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import sys
import warnings
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.dataset import _repr, files_exist

from dgrag.kg_construction.utils import KG_DELIMITER
from dgrag.text_emb_models import BaseTextEmbModel
from dgrag.utils import get_rank

logger = logging.getLogger(__name__)


def atomic_torch_save(obj: Any, path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def build_stage2_fingerprint(
    text_emb_model_cfgs: DictConfig,
    compute_structural_score: bool = False,
    structural_score_alpha: float = 0.5,
) -> str:
    payload: dict[str, Any] = {
        "text_emb_model_cfgs": OmegaConf.to_container(
            text_emb_model_cfgs, resolve=True
        ),
        "compute_structural_score": compute_structural_score,
        "structural_score_alpha": structural_score_alpha,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class KGDataset(InMemoryDataset):
    """A dataset class for processing and managing Knowledge Graph (KG) data."""

    delimiter = KG_DELIMITER

    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_cfgs: DictConfig,
        force_rebuild: bool = False,
        compute_structural_score: bool = False,
        structural_score_alpha: float = 0.5,
        **kwargs: str,
    ) -> None:
        self.name = data_name
        self.force_rebuild = force_rebuild
        self.compute_structural_score = compute_structural_score
        self.structural_score_alpha = structural_score_alpha
        self.fingerprint = build_stage2_fingerprint(
            text_emb_model_cfgs=text_emb_model_cfgs,
            compute_structural_score=compute_structural_score,
            structural_score_alpha=structural_score_alpha,
        )
        self.text_emb_model_cfgs = text_emb_model_cfgs
        super().__init__(root, None, None)
        try:
            self.data, self.slices = torch.load(
                self.processed_paths[0], weights_only=False
            )
        except (EOFError, RuntimeError, pickle.UnpicklingError) as exc:
            logger.warning(
                "Processed KG cache for %s is unreadable at %s: %s. Rebuilding.",
                self.name,
                self.processed_paths[0],
                exc,
            )
            self.force_rebuild = True
            self._process()
            self.data, self.slices = torch.load(
                self.processed_paths[0], weights_only=False
            )
        self.feat_dim = self._data.rel_emb.size(1)

    @property
    def raw_file_names(self) -> list:
        return ["kg.txt"]

    def load_file(
        self, triplet_file: str, inv_entity_vocab: dict, inv_rel_vocab: dict
    ) -> dict:
        triplets = []
        entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)

        with open(triplet_file, encoding="utf-8") as fin:
            for line in fin:
                try:
                    u, r, v = (
                        line.split()
                        if self.delimiter is None
                        else line.strip().split(self.delimiter)
                    )
                except Exception as e:
                    logger.error(f"Error in line: {line}, {e}, Skipping")
                    continue
                if u not in inv_entity_vocab:
                    inv_entity_vocab[u] = entity_cnt
                    entity_cnt += 1
                if v not in inv_entity_vocab:
                    inv_entity_vocab[v] = entity_cnt
                    entity_cnt += 1
                if r not in inv_rel_vocab:
                    inv_rel_vocab[r] = rel_cnt
                    rel_cnt += 1
                u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]
                triplets.append((u, v, r))

        return {
            "triplets": triplets,
            "num_node": len(inv_entity_vocab),
            "num_relation": rel_cnt,
            "inv_entity_vocab": inv_entity_vocab,
            "inv_rel_vocab": inv_rel_vocab,
        }

    def _process(self) -> None:
        f = osp.join(self.processed_dir, "pre_transform.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_transform
        ):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first",
                stacklevel=1,
            )

        f = osp.join(self.processed_dir, "pre_filter.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_filter
        ):
            warnings.warn(
                f"The `pre_filter` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-fitering technique, make sure to "
                f"delete '{self.processed_dir}' first",
                stacklevel=1,
            )

        if self.force_rebuild or not files_exist(self.processed_paths):
            logger.warning(f"Processing KG dataset {self.name} at rank {get_rank()}")
            if self.log and "pytest" not in sys.modules:
                print("Processing...", file=sys.stderr)

            os.makedirs(self.processed_dir, exist_ok=True)
            self.process()

            path = osp.join(self.processed_dir, "pre_transform.pt")
            atomic_torch_save(_repr(self.pre_transform), path)
            path = osp.join(self.processed_dir, "pre_filter.pt")
            atomic_torch_save(_repr(self.pre_filter), path)

            if self.log and "pytest" not in sys.modules:
                print("Done!", file=sys.stderr)

    def process(self) -> None:
        kg_file = self.raw_paths[0]
        kg_result = self.load_file(kg_file, inv_entity_vocab={}, inv_rel_vocab={})

        num_node = kg_result["num_node"]
        num_relations = kg_result["num_relation"]
        kg_triplets = kg_result["triplets"]

        train_target_edges = torch.tensor(
            [[t[0], t[1]] for t in kg_triplets], dtype=torch.long
        ).t()
        train_target_etypes = torch.tensor([t[2] for t in kg_triplets])

        train_edges = torch.cat([train_target_edges, train_target_edges.flip(0)], dim=1)
        train_etypes = torch.cat(
            [train_target_etypes, train_target_etypes + num_relations]
        )

        with open(self.processed_dir + "/ent2id.json", "w") as f:
            json.dump(kg_result["inv_entity_vocab"], f)
        rel2id = kg_result["inv_rel_vocab"]
        id2rel = {v: k for k, v in rel2id.items()}
        for etype in train_etypes:
            if etype.item() >= num_relations:
                raw_etype = etype - num_relations
                raw_rel = id2rel[raw_etype.item()]
                rel2id["inverse_" + raw_rel] = etype.item()
        with open(self.processed_dir + "/rel2id.json", "w") as f:
            json.dump(rel2id, f)

        logger.info("Generating relation embeddings")
        text_emb_model: BaseTextEmbModel = instantiate(self.text_emb_model_cfgs)
        rel_emb = text_emb_model.encode(list(rel2id.keys()), is_query=False).cpu()

        kg_data = Data(
            edge_index=train_edges,
            edge_type=train_etypes,
            num_nodes=num_node,
            target_edge_index=train_target_edges,
            target_edge_type=train_target_etypes,
            num_relations=num_relations * 2,
            rel_emb=rel_emb,
        )
        if self.compute_structural_score:
            kg_data.structure_score = self._compute_structure_score(
                num_nodes=num_node,
                target_edge_index=train_target_edges,
                alpha=self.structural_score_alpha,
            )

        atomic_torch_save((self.collate([kg_data])), self.processed_paths[0])

        with open(self.processed_dir + "/text_emb_model_cfgs.json", "w") as f:
            json.dump(OmegaConf.to_container(self.text_emb_model_cfgs), f, indent=4)

    def _compute_structure_score(
        self,
        num_nodes: int,
        target_edge_index: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from(target_edge_index.t().tolist())
        graph.remove_edges_from(nx.selfloop_edges(graph))

        core_number = nx.core_number(graph) if graph.number_of_edges() > 0 else {}
        clustering = nx.clustering(graph)

        score = torch.zeros(num_nodes, dtype=torch.float32)
        for node_idx in range(num_nodes):
            core_indicator = 1.0 if core_number.get(node_idx, 0) >= 2 else 0.0
            clustering_score = float(clustering.get(node_idx, 0.0))
            score[node_idx] = alpha * core_indicator + (1 - alpha) * clustering_score
        return score.clamp_(0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.name}()"

    @property
    def num_relations(self) -> int:
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self) -> str:
        return os.path.join(str(self.root), str(self.name), "processed", "stage1")

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    @property
    def processed_dir(self) -> str:
        return os.path.join(
            str(self.root),
            str(self.name),
            "processed",
            "stage2",
            self.fingerprint,
        )
