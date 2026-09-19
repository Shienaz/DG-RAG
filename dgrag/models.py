from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data

from dgrag.losses import BCELoss, ListCELoss, QuestionEntityContrastiveLoss
from dgrag.ultra.models import EntityNBFNet, QueryNBFNet


class QueryGNN(nn.Module):
    def __init__(
        self, entity_model: EntityNBFNet, rel_emb_dim: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__()
        self.rel_emb_dim = rel_emb_dim
        self.entity_model = entity_model
        self.rel_mlp = nn.Linear(rel_emb_dim, self.entity_model.dims[0])
        self._last_auxiliary_loss: torch.Tensor | None = None
        self._last_auxiliary_stats: dict[str, float] = {}

    def forward(self, data: Data, batch: torch.Tensor) -> torch.Tensor:
        batch_size = len(batch)
        relation_representations = (
            self.rel_mlp(data.rel_emb).unsqueeze(0).expand(batch_size, -1, -1)
        )
        h_index, t_index, r_index = batch.unbind(-1)
        data = self.entity_model.remove_easy_edges(data, h_index, t_index, r_index)
        score = self.entity_model(data, relation_representations, batch)
        self._last_auxiliary_loss = self.entity_model.get_auxiliary_loss()
        if hasattr(self.entity_model, "get_auxiliary_stats_dict"):
            self._last_auxiliary_stats = self.entity_model.get_auxiliary_stats_dict()
        else:
            self._last_auxiliary_stats = {}
        return score

    def get_auxiliary_loss(self) -> torch.Tensor | None:
        return self._last_auxiliary_loss

    def get_auxiliary_stats_dict(self) -> dict[str, float]:
        return dict(self._last_auxiliary_stats)


class GNNRetriever(QueryGNN):
    def __init__(
        self,
        entity_model: QueryNBFNet,
        rel_emb_dim: int,
        use_question_entity_contrastive: bool = False,
        question_entity_contrastive_weight: float = 0.05,
        question_entity_contrastive_temperature: float = 0.2,
        question_entity_contrastive_mode: str = "batch",
        question_entity_hard_negative_topk: int = 256,
        question_entity_hard_negative_exclude_question_entities: bool = True,
        question_entity_hard_negative_random_fallback: bool = True,
        use_question_relation_adapter: bool = False,
        question_relation_adapter_hidden_dim: int | None = None,
        question_relation_adapter_scale: float = 0.1,
        branch_supervised_weight: float = 0.0,
        branch_supervised_bce_weight: float = 0.3,
        branch_supervised_listce_weight: float = 0.7,
        branch_supervised_adversarial_temperature: float = 0.2,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(entity_model, rel_emb_dim)
        self.question_mlp = nn.Linear(self.rel_emb_dim, self.entity_model.dims[0])
        self.use_question_entity_contrastive = use_question_entity_contrastive
        self.question_entity_contrastive_weight = question_entity_contrastive_weight
        if question_entity_contrastive_mode not in {"batch", "hard", "hybrid"}:
            raise ValueError(
                "question_entity_contrastive_mode must be 'batch', 'hard', or 'hybrid'"
            )
        if question_entity_hard_negative_topk < 1:
            raise ValueError("question_entity_hard_negative_topk must be positive")
        self.question_entity_contrastive_mode = question_entity_contrastive_mode
        self.question_entity_hard_negative_topk = question_entity_hard_negative_topk
        self.question_entity_hard_negative_exclude_question_entities = (
            question_entity_hard_negative_exclude_question_entities
        )
        self.question_entity_hard_negative_random_fallback = (
            question_entity_hard_negative_random_fallback
        )
        self.use_question_relation_adapter = use_question_relation_adapter
        self.question_relation_adapter_scale = question_relation_adapter_scale
        self.branch_supervised_weight = branch_supervised_weight
        self.branch_supervised_bce_weight = branch_supervised_bce_weight
        self.branch_supervised_listce_weight = branch_supervised_listce_weight
        self._last_question_entity_contrastive_loss: float | None = None
        self._last_question_entity_hard_negative_count: float | None = None
        self._last_branch_supervised_loss: float | None = None
        self._last_external_decision_mutual_loss: float | None = None
        self._last_decision_mutual_candidate_count: float | None = None

        branch_geometries = list(getattr(self.entity_model, "branch_geometries", []) or [])
        dual_branch = branch_geometries in (
            ["lorentz", "euclidean"], ["euclidean", "euclidean"]
        )
        if self.branch_supervised_weight > 0:
            if not dual_branch:
                raise ValueError(
                    "Branch supervised loss requires Lo+Eu or Dual-Euclidean branches"
                )
            if getattr(self.entity_model, "head_h", None) is None or getattr(
                self.entity_model, "head_e", None
            ) is None:
                if not hasattr(self.entity_model, "_build_mlp"):
                    raise ValueError(
                        "Branch supervised loss requires an entity model that can build branch heads"
                    )
                branch_feature_dim = self.entity_model.dims[0] * 2
                self.entity_model.head_h = self.entity_model._build_mlp(
                    branch_feature_dim
                )
                self.entity_model.head_e = self.entity_model._build_mlp(
                    branch_feature_dim
                )
            if hasattr(self.entity_model, "needs_branch_supervision"):
                self.entity_model.needs_branch_supervision = True
        if self.use_question_entity_contrastive:
            if branch_geometries != ["lorentz"] and not dual_branch:
                raise ValueError(
                    "Question-entity contrastive loss requires Lorentz, Lo+Eu, "
                    "or Dual-Euclidean branches"
                )
        if self.use_question_relation_adapter:
            if branch_geometries != ["lorentz"]:
                raise ValueError(
                    "Question-relation adapter requires branch_geometries=['lorentz']"
                )

        if self.use_question_entity_contrastive:
            self.question_align_proj = nn.Linear(
                self.entity_model.dims[0], self.entity_model.dims[0]
            )
            self.question_entity_node_proj = (
                nn.Linear(self.entity_model.dims[0] * 2, self.entity_model.dims[0])
                if dual_branch
                else None
            )
            self.question_entity_contrastive_loss = QuestionEntityContrastiveLoss(
                temperature=question_entity_contrastive_temperature
            )
        else:
            self.question_align_proj = None
            self.question_entity_node_proj = None
            self.question_entity_contrastive_loss = None

        if self.use_question_relation_adapter:
            hidden_dim = (
                question_relation_adapter_hidden_dim
                if question_relation_adapter_hidden_dim is not None
                else self.entity_model.dims[0]
            )
            self.relation_adapter = nn.Sequential(
                nn.Linear(self.entity_model.dims[0] * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.entity_model.dims[0]),
            )
        else:
            self.relation_adapter = None

        self.branch_bce_loss = BCELoss(
            adversarial_temperature=branch_supervised_adversarial_temperature
        )
        self.branch_listce_loss = ListCELoss()

    def _build_relation_representations(
        self, graph: Data, question_embedding: torch.Tensor
    ) -> torch.Tensor:
        relation_representations = self.rel_mlp(graph.rel_emb)
        relation_representations = relation_representations.unsqueeze(0).expand(
            question_embedding.size(0), -1, -1
        )
        if self.relation_adapter is None:
            return relation_representations

        question_relation = question_embedding.unsqueeze(1).expand(
            -1, relation_representations.size(1), -1
        )
        relation_delta = self.relation_adapter(
            torch.cat([question_relation, relation_representations], dim=-1)
        )
        return relation_representations + (
            self.question_relation_adapter_scale * relation_delta
        )

    def _compute_question_entity_auxiliary_loss(
        self,
        batch: dict[str, torch.Tensor],
        question_embedding: torch.Tensor,
        candidate_scores: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            not self.use_question_entity_contrastive
            or self.question_align_proj is None
            or self.question_entity_contrastive_loss is None
        ):
            return None
        if "supporting_entities_masks" not in batch:
            raise ValueError(
                "supporting_entities_masks is required when question-entity contrastive loss is enabled"
            )
        if hasattr(self.entity_model, "get_question_entity_node_repr"):
            node_repr = self.entity_model.get_question_entity_node_repr()
        elif hasattr(self.entity_model, "get_last_node_tangent"):
            node_repr = self.entity_model.get_last_node_tangent()
        else:
            raise ValueError(
                "Question-entity contrastive loss requires an entity model exposing "
                "get_question_entity_node_repr() or get_last_node_tangent()"
            )

        if node_repr is None:
            raise RuntimeError(
                "Entity model did not expose final node states for question-entity contrastive loss"
            )
        if self.question_entity_node_proj is not None:
            node_repr = self.question_entity_node_proj(node_repr)
        elif node_repr.size(-1) != self.entity_model.dims[0]:
            raise ValueError(
                "Question-entity node representation has incompatible dimension "
                f"{node_repr.size(-1)}; expected {self.entity_model.dims[0]}"
            )

        question_repr = self.question_align_proj(question_embedding)
        contrastive_losses = []
        if self.question_entity_contrastive_mode in {"batch", "hybrid"}:
            contrastive_losses.append(
                self.question_entity_contrastive_loss(
                    question_repr,
                    node_repr,
                    batch["supporting_entities_masks"],
                )
            )
        if self.question_entity_contrastive_mode in {"hard", "hybrid"}:
            negative_mask = self._build_question_entity_hard_negative_mask(
                batch,
                candidate_scores,
            )
            contrastive_losses.append(
                self.question_entity_contrastive_loss(
                    question_repr,
                    node_repr,
                    batch["supporting_entities_masks"],
                    negative_mask=negative_mask,
                )
            )
            self._last_question_entity_hard_negative_count = float(
                self.question_entity_contrastive_loss.last_hard_negative_count
            )
        contrastive_loss = sum(contrastive_losses)
        weighted_loss = self.question_entity_contrastive_weight * contrastive_loss
        self._last_question_entity_contrastive_loss = float(
            weighted_loss.detach().item()
        )
        return weighted_loss

    def _build_question_entity_hard_negative_mask(
        self,
        batch: dict[str, torch.Tensor],
        candidate_scores: torch.Tensor,
    ) -> torch.Tensor:
        positive_mask = batch["supporting_entities_masks"].to(
            device=candidate_scores.device
        ).bool()
        eligible_mask = ~positive_mask
        if self.question_entity_hard_negative_exclude_question_entities:
            eligible_mask = eligible_mask & ~batch["question_entities_masks"].to(
                device=candidate_scores.device
            ).bool()

        k = min(self.question_entity_hard_negative_topk, candidate_scores.size(-1))
        masked_scores = candidate_scores.detach().masked_fill(
            ~eligible_mask, torch.finfo(candidate_scores.dtype).min
        )
        top_index = masked_scores.topk(k, dim=-1).indices
        negative_mask = torch.zeros_like(eligible_mask)
        negative_mask.scatter_(1, top_index, True)
        negative_mask = negative_mask & eligible_mask

        missing_negative = ~negative_mask.any(dim=-1)
        if (
            self.question_entity_hard_negative_random_fallback
            and bool(missing_negative.any())
        ):
            random_scores = torch.rand_like(candidate_scores).masked_fill(
                ~eligible_mask, torch.finfo(candidate_scores.dtype).min
            )
            fallback_index = random_scores.topk(1, dim=-1).indices
            fallback_mask = torch.zeros_like(eligible_mask)
            fallback_mask.scatter_(1, fallback_index, True)
            negative_mask = negative_mask | (
                fallback_mask & eligible_mask & missing_negative.unsqueeze(-1)
            )
        return negative_mask

    def _compute_branch_supervised_loss(
        self,
        branch_scores: dict[str, torch.Tensor] | None,
        target: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.branch_supervised_weight <= 0:
            return None
        if branch_scores is None or "h" not in branch_scores or "e" not in branch_scores:
            raise RuntimeError("Branch scores are required for branch supervised loss")

        score_h = branch_scores["h"]
        score_e = branch_scores["e"]
        target = target.to(device=score_h.device, dtype=score_h.dtype)
        loss_h = (
            self.branch_supervised_bce_weight * self.branch_bce_loss(score_h, target)
            + self.branch_supervised_listce_weight
            * self.branch_listce_loss(score_h, target)
        )
        loss_e = (
            self.branch_supervised_bce_weight * self.branch_bce_loss(score_e, target)
            + self.branch_supervised_listce_weight
            * self.branch_listce_loss(score_e, target)
        )
        weighted_loss = self.branch_supervised_weight * 0.5 * (loss_h + loss_e)
        self._last_branch_supervised_loss = float(weighted_loss.detach().item())
        return weighted_loss

    def _build_decision_mutual_candidate_mask(
        self,
        branch_scores: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        candidate_mode = getattr(
            self.entity_model, "decision_mutual_candidate_mode", "all"
        )
        if candidate_mode != "hard":
            return None
        if (
            "supporting_entities_masks" not in batch
            or "question_entities_masks" not in batch
        ):
            raise ValueError(
                "Hard-candidate DML requires supporting_entities_masks and question_entities_masks"
            )

        score_fuse = branch_scores["fuse"]
        candidate_mask = (
            batch["supporting_entities_masks"].to(device=score_fuse.device).bool()
            | batch["question_entities_masks"].to(device=score_fuse.device).bool()
        )
        topk = min(int(getattr(self.entity_model, "decision_mutual_topk", 256)), score_fuse.size(-1))
        top_index = score_fuse.detach().topk(topk, dim=-1).indices
        candidate_mask = candidate_mask.clone()
        candidate_mask.scatter_(1, top_index, True)
        self._last_decision_mutual_candidate_count = float(
            candidate_mask.sum(dim=-1).float().mean().detach().item()
        )
        return candidate_mask

    def _compute_external_decision_mutual_loss(
        self,
        branch_scores: dict[str, torch.Tensor] | None,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if branch_scores is None:
            return None
        if not bool(getattr(self.entity_model, "use_decision_mutual_loss", False)):
            return None
        if not self.training:
            return None
        if getattr(self.entity_model, "decision_mutual_candidate_mode", "all") != "hard":
            return None
        if not hasattr(self.entity_model, "decision_mutual_loss_fn"):
            return None

        candidate_mask = self._build_decision_mutual_candidate_mask(branch_scores, batch)
        teacher_score = (
            branch_scores["fuse"]
            if getattr(self.entity_model, "decision_mutual_teacher", "peer") == "fuse"
            else None
        )
        raw_loss = self.entity_model.decision_mutual_loss_fn(
            branch_scores["h"],
            branch_scores["e"],
            teacher_score=teacher_score,
            candidate_mask=candidate_mask,
            branch_kl_weight=getattr(
                self.entity_model, "decision_mutual_branch_kl_weight", 0.2
            ),
        )
        weighted_loss = getattr(self.entity_model, "decision_mutual_weight", 1.0) * raw_loss
        self._last_external_decision_mutual_loss = float(
            weighted_loss.detach().item()
        )
        return weighted_loss

    def forward(
        self,
        graph: Data,
        batch: dict[str, torch.Tensor],
        entities_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        question_emb = batch["question_embeddings"]
        question_entities_mask = batch["question_entities_masks"]

        question_embedding = self.question_mlp(question_emb)
        relation_representations = self._build_relation_representations(
            graph, question_embedding
        )

        if entities_weight is not None:
            question_entities_mask = question_entities_mask * entities_weight.unsqueeze(0)

        input_tensor = torch.einsum(
            "bn, bd -> bnd", question_entities_mask, question_embedding
        )

        output = self.entity_model(
            graph, input_tensor, relation_representations, question_embedding
        )
        entity_auxiliary_loss = self.entity_model.get_auxiliary_loss()
        self._last_question_entity_contrastive_loss = None
        self._last_question_entity_hard_negative_count = None
        self._last_branch_supervised_loss = None
        self._last_external_decision_mutual_loss = None
        self._last_decision_mutual_candidate_count = None
        if hasattr(self.entity_model, "get_last_branch_scores"):
            branch_scores = self.entity_model.get_last_branch_scores()
        else:
            branch_scores = None
        branch_supervised_loss = None
        if self.branch_supervised_weight > 0:
            if "supporting_entities_masks" not in batch:
                raise ValueError(
                    "supporting_entities_masks is required for branch supervised loss"
                )
            branch_supervised_loss = self._compute_branch_supervised_loss(
                branch_scores,
                batch["supporting_entities_masks"],
            )
        external_decision_mutual_loss = self._compute_external_decision_mutual_loss(
            branch_scores,
            batch,
        )
        question_entity_auxiliary_loss = self._compute_question_entity_auxiliary_loss(
            batch, question_embedding, output
        )
        auxiliary_losses = [
            loss
            for loss in (
                entity_auxiliary_loss,
                branch_supervised_loss,
                external_decision_mutual_loss,
                question_entity_auxiliary_loss,
            )
            if loss is not None
        ]
        self._last_auxiliary_loss = (
            sum(auxiliary_losses) if auxiliary_losses else None
        )

        if hasattr(self.entity_model, "get_auxiliary_stats_dict"):
            self._last_auxiliary_stats = self.entity_model.get_auxiliary_stats_dict()
        else:
            self._last_auxiliary_stats = {}
        if self._last_branch_supervised_loss is not None:
            self._last_auxiliary_stats["branch_supervised_loss"] = (
                self._last_branch_supervised_loss
            )
        if self._last_external_decision_mutual_loss is not None:
            self._last_auxiliary_stats["decision_mutual_loss"] = (
                self._last_external_decision_mutual_loss
            )
        if self._last_decision_mutual_candidate_count is not None:
            self._last_auxiliary_stats["decision_mutual_candidate_count"] = (
                self._last_decision_mutual_candidate_count
            )
        if self._last_question_entity_contrastive_loss is not None:
            self._last_auxiliary_stats["question_entity_contrastive_loss"] = (
                self._last_question_entity_contrastive_loss
            )
            if self.question_entity_contrastive_loss is not None:
                self._last_auxiliary_stats[
                    "question_entity_contrastive_global_batch_size"
                ] = float(
                    getattr(
                        self.question_entity_contrastive_loss,
                        "last_global_batch_size",
                        0,
                    )
                )
                self._last_auxiliary_stats[
                    "question_entity_contrastive_local_valid_size"
                ] = float(
                    getattr(
                        self.question_entity_contrastive_loss,
                        "last_local_valid_size",
                        0,
                    )
                )
                self._last_auxiliary_stats[
                    "question_entity_contrastive_hard_negative_count"
                ] = float(
                    getattr(
                        self.question_entity_contrastive_loss,
                        "last_hard_negative_count",
                        0,
                    )
                )

        return output

    def visualize(
        self,
        graph: Data,
        sample: dict[str, torch.Tensor],
        entities_weight: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        question_emb = sample["question_embeddings"]
        question_entities_mask = sample["question_entities_masks"]
        question_embedding = self.question_mlp(question_emb)
        batch_size = question_embedding.size(0)

        assert batch_size == 1, "Currently only supports batch size 1 for visualization"

        relation_representations = self._build_relation_representations(
            graph, question_embedding
        )

        if entities_weight is not None:
            question_entities_mask = question_entities_mask * entities_weight.unsqueeze(0)

        input_tensor = torch.einsum(
            "bn, bd -> bnd", question_entities_mask, question_embedding
        )
        return self.entity_model.visualize(
            graph,
            sample,
            input_tensor,
            relation_representations,
            question_embedding,
        )
