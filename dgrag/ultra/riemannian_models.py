from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from dgrag.losses import (
    BranchDecisionMutualLoss,
    GeometryAlignmentLoss,
    TopologyMutualLearningLoss,
    TopologySpecializationLoss,
)
from dgrag.ultra.base_nbfnet import BaseNBFNet
from dgrag.ultra.geometry import Euclidean, Lorentz
from dgrag.ultra.layers import GeneralizedRelationalConv
from dgrag.ultra.riemannian_layers import (
    EuclideanAttentionRelationalConv,
    RiemannianRelationalConv,
    _clip_tangent_norm,
)


class _BaseRiemannianNBFNet(BaseNBFNet):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] | int,
        num_relation: int = 1,
        message_func: str = "distmult",
        aggregate_func: str = "attention_frechet",
        short_cut: bool = True,
        layer_norm: bool = False,
        activation: str = "relu",
        concat_hidden: bool = False,
        num_mlp_layer: int = 2,
        branch_geometries: Sequence[str] | None = None,
        use_alignment_loss: bool = True,
        alignment_weight: float = 0.1,
        alignment_temperature: float = 0.2,
        alignment_sample_size: int = 128,
        use_decision_mutual_loss: bool = False,
        decision_mutual_weight: float = 0.01,
        decision_mutual_temperature: float = 1.0,
        decision_mutual_divergence: str = "kl",
        decision_mutual_teacher: str = "peer",
        decision_mutual_branch_kl_weight: float = 0.2,
        decision_mutual_candidate_mode: str = "all",
        decision_mutual_topk: int = 256,
        needs_branch_supervision: bool = False,
        use_topology_mutual_loss: bool = False,
        topology_mutual_weight: float = 0.005,
        topology_mutual_temperature: float = 0.5,
        topology_mutual_divergence: str = "kl",
        topology_mutual_sample_size: int = 64,
        topology_kernel_degree: int = 2,
        topology_kernel_bias: float = 0.0,
        topology_mask_self: bool = True,
        use_structure_gated_readout: bool = False,
        specialization_weight: float = 0.05,
        specialization_margin: float = 0.1,
        gate_tau_low: float = 0.1,
        gate_tau_high: float = 0.75,
        gate_beta: float = 0.1,
        learnable_curvature: bool = False,
        boundary_max_tangent_norm: float | None = 10.0,
        message_max_tangent_norm: float | None = 10.0,
        update_max_tangent_norm: float | None = 10.0,
        use_edge_parallel: bool = False,
        edge_parallel_shard_strategy: str = "contiguous",
        euclidean_edge_weight_requires_grad: bool = True,
        euclidean_attention_rspmm: bool = True,
        dual_euclidean_branch_one: str = "plain",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_relation=num_relation,
            message_func=message_func,
            aggregate_func=aggregate_func,
            short_cut=short_cut,
            layer_norm=layer_norm,
            activation=activation,
            concat_hidden=concat_hidden,
            num_mlp_layer=num_mlp_layer,
            **kwargs,
        )
        if len(set(self.dims)) != 1:
            raise ValueError(
                "Riemannian models currently require input_dim and hidden_dims to match"
            )
        if not (0 < gate_tau_low < gate_tau_high < 1):
            raise ValueError("gate_tau_low and gate_tau_high must satisfy 0 < low < high < 1")
        if decision_mutual_teacher not in {"peer", "fuse"}:
            raise ValueError("decision_mutual_teacher must be 'peer' or 'fuse'")
        if decision_mutual_candidate_mode not in {"all", "hard"}:
            raise ValueError("decision_mutual_candidate_mode must be 'all' or 'hard'")
        if decision_mutual_branch_kl_weight < 0:
            raise ValueError("decision_mutual_branch_kl_weight must be non-negative")
        if decision_mutual_topk < 1:
            raise ValueError("decision_mutual_topk must be positive")

        self.branch_geometries = list(branch_geometries or ["lorentz", "euclidean"])
        if self.branch_geometries not in (
            ["lorentz"], ["lorentz", "euclidean"], ["euclidean", "euclidean"]
        ):
            raise ValueError(
                "Supported branches: ['lorentz'], ['lorentz', 'euclidean'], "
                "or ['euclidean', 'euclidean']"
            )
        self.is_dual_euclidean = self.branch_geometries == ["euclidean", "euclidean"]
        if self.is_dual_euclidean and learnable_curvature:
            raise ValueError("Dual-Euclidean requires learnable_curvature=false")
        self.dual_euclidean_branch_one = str(dual_euclidean_branch_one).lower()
        if self.dual_euclidean_branch_one not in ("plain", "attention"):
            raise ValueError(
                "dual_euclidean_branch_one must be 'plain' or 'attention'"
            )
        # 'plain' gives branch 1 the unmodified Euclidean layer that the Eu-only
        # protocol builds in `dgrag.ultra.models.EntityNBFNet`, so the two
        # branches are structural twins and both ride the fused rspmm kernel.
        # 'attention' keeps the legacy parameter-matched
        # EuclideanAttentionRelationalConv control, which is not a twin.
        self.use_plain_euclidean_branch_one = (
            self.is_dual_euclidean and self.dual_euclidean_branch_one == "plain"
        )
        self.use_euclidean_branch = "euclidean" in self.branch_geometries
        self.num_branches = len(self.branch_geometries)
        self.hidden_dim = self.dims[0]
        self.manifold_h = (
            Euclidean() if self.is_dual_euclidean
            else Lorentz(learnable=learnable_curvature)
        )

        self.use_alignment_loss = use_alignment_loss and self.use_euclidean_branch
        self.alignment_weight = alignment_weight
        self.alignment_sample_size = alignment_sample_size
        self.use_decision_mutual_loss = (
            use_decision_mutual_loss and self.use_euclidean_branch
        )
        self.decision_mutual_weight = decision_mutual_weight
        self.decision_mutual_temperature = decision_mutual_temperature
        self.decision_mutual_divergence = decision_mutual_divergence
        self.decision_mutual_teacher = decision_mutual_teacher
        self.decision_mutual_branch_kl_weight = decision_mutual_branch_kl_weight
        self.decision_mutual_candidate_mode = decision_mutual_candidate_mode
        self.decision_mutual_topk = decision_mutual_topk
        self.needs_branch_supervision = (
            needs_branch_supervision and self.use_euclidean_branch
        )
        self.use_topology_mutual_loss = (
            use_topology_mutual_loss and self.use_euclidean_branch
        )
        self.topology_mutual_weight = topology_mutual_weight
        self.topology_mutual_divergence = topology_mutual_divergence
        self.boundary_max_tangent_norm = boundary_max_tangent_norm
        self.use_edge_parallel = use_edge_parallel
        self.edge_parallel_shard_strategy = edge_parallel_shard_strategy
        self.euclidean_edge_weight_requires_grad = euclidean_edge_weight_requires_grad
        self.use_structure_gated_readout = (
            use_structure_gated_readout and self.use_euclidean_branch
        )
        self.specialization_weight = specialization_weight
        self.gate_tau_low = gate_tau_low
        self.gate_tau_high = gate_tau_high
        self.gate_beta = gate_beta

        self.alignment_loss_fn = GeometryAlignmentLoss(temperature=alignment_temperature)
        self.decision_mutual_loss_fn = BranchDecisionMutualLoss(
            temperature=decision_mutual_temperature,
            divergence=decision_mutual_divergence,
        )
        self.topology_mutual_loss_fn = TopologyMutualLearningLoss(
            temperature=topology_mutual_temperature,
            sample_size=topology_mutual_sample_size,
            kernel_degree=topology_kernel_degree,
            kernel_bias=topology_kernel_bias,
            mask_self=topology_mask_self,
            divergence=topology_mutual_divergence,
        )
        self.specialization_loss_fn = TopologySpecializationLoss(
            margin=specialization_margin
        )
        self._last_alignment_loss: torch.Tensor | None = None
        self._last_decision_mutual_loss: torch.Tensor | None = None
        self._last_topology_mutual_loss: torch.Tensor | None = None
        self._last_specialization_loss: torch.Tensor | None = None
        self._last_auxiliary_stats: dict[str, float] = {}

        if self.use_plain_euclidean_branch_one:
            first_stack = self._build_plain_euclidean_stack(
                num_relation,
                layer_norm,
                activation,
                use_edge_parallel,
                edge_parallel_shard_strategy,
            )
        else:
            first_branch_layer = (
                EuclideanAttentionRelationalConv if self.is_dual_euclidean
                else RiemannianRelationalConv
            )
            first_branch_kwargs: dict[str, Any] = dict(
                input_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                query_input_dim=self.hidden_dim,
                branch_geometries=[
                    "euclidean" if self.is_dual_euclidean else "lorentz"
                ],
                message_func=message_func,
                aggregate_func=aggregate_func,
                layer_norm=layer_norm,
                activation=activation,
                residual_in_tangent_space=short_cut,
                learnable_curvature=learnable_curvature,
                boundary_max_tangent_norm=boundary_max_tangent_norm,
                message_max_tangent_norm=message_max_tangent_norm,
                update_max_tangent_norm=update_max_tangent_norm,
                use_edge_parallel=use_edge_parallel,
                edge_parallel_shard_strategy=edge_parallel_shard_strategy,
            )
            if self.is_dual_euclidean:
                # Only the flat attention reduce can be fused; the Lorentz branch
                # keeps its tangent-space Frechet midpoint.
                first_branch_kwargs["use_rspmm_aggregation"] = (
                    euclidean_attention_rspmm
                )
            first_stack = [
                first_branch_layer(**first_branch_kwargs)
                for _ in range(len(self.dims) - 1)
            ]
        self.layers_h = nn.ModuleList(first_stack)
        self.layers_e = nn.ModuleList(
            self._build_plain_euclidean_stack(
                num_relation,
                layer_norm,
                activation,
                use_edge_parallel,
                edge_parallel_shard_strategy,
            )
        ) if self.use_euclidean_branch else None

        hidden_dim_list = list(hidden_dims) if isinstance(hidden_dims, Sequence) else [hidden_dims]
        branch_feature_dim = (
            self.num_branches * sum(hidden_dim_list)
            if self.concat_hidden
            else self.num_branches * hidden_dim_list[-1]
        )
        fused_feature_dim = branch_feature_dim + input_dim
        self.mlp = self._build_mlp(fused_feature_dim)
        if (
            self.use_structure_gated_readout
            or self.use_decision_mutual_loss
            or self.needs_branch_supervision
        ):
            self.head_h = self._build_mlp(hidden_dim_list[-1] + input_dim)
            self.head_e = self._build_mlp(hidden_dim_list[-1] + input_dim)
        else:
            self.head_h = None
            self.head_e = None

        if self.use_alignment_loss:
            self.align_h_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.align_e_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.align_h_proj = None
            self.align_e_proj = None

    def _build_plain_euclidean_stack(
        self,
        num_relation: int,
        layer_norm: bool,
        activation: Any,
        use_edge_parallel: bool,
        edge_parallel_shard_strategy: str,
    ) -> list[nn.Module]:
        """One stack of the unmodified Euclidean NBFNet relation layer.

        This is byte-for-byte the layer `dgrag.ultra.models.EntityNBFNet` uses
        for the Eu-only protocol (`distmult` messages, plain `sum` aggregation,
        projected relations, no curvature, no clipping). `layers_e` has always
        been built this way; `dual_euclidean_branch_one="plain"` gives branch 1
        an exact structural twin of it, so both branches reduce through the same
        fused ``generalized_rspmm`` path whenever ``edge_weight`` carries no
        gradient (the production setting).
        """
        return [
            GeneralizedRelationalConv(
                self.hidden_dim,
                self.hidden_dim,
                num_relation,
                self.hidden_dim,
                message_func="distmult",
                aggregate_func="sum",
                layer_norm=layer_norm,
                activation=activation,
                dependent=False,
                project_relations=True,
                use_edge_parallel=use_edge_parallel,
                edge_parallel_shard_strategy=edge_parallel_shard_strategy,
            )
            for _ in range(len(self.dims) - 1)
        ]

    def _bind_relations(self, relation_representations: torch.Tensor) -> None:
        """Rebind per-layer relation features before message passing.

        `GeneralizedRelationalConv` reads `self.relation` inside its own
        `forward`, so every plain-Euclidean stack must be rebound on each call.
        The Riemannian/attention stack receives relations as a call argument
        instead, so this is a no-op for it.

        Order matters only for reproducibility: `layers_h` is bound before
        `layers_e`, matching the construction order.
        """
        if self.use_plain_euclidean_branch_one:
            for layer in self.layers_h:
                layer.relation = relation_representations
        if self.use_euclidean_branch and self.layers_e is not None:
            for layer in self.layers_e:
                layer.relation = relation_representations

    def _build_mlp(self, feature_dim: int) -> nn.Sequential:
        mlp_layers: list[nn.Module] = []
        for _ in range(self.num_mlp_layers - 1):
            mlp_layers.append(nn.Linear(feature_dim, feature_dim))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(feature_dim, 1))
        return nn.Sequential(*mlp_layers)

    def _expmap_branch(self, manifold: Lorentz | Euclidean, tangent: torch.Tensor) -> torch.Tensor:
        flat = tangent.reshape(-1, tangent.size(-1))
        flat = manifold.proju0(flat)
        return manifold.expmap0(flat).reshape_as(tangent)

    def _sample_alignment_features(
        self, h_tangent: torch.Tensor, e_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_flat = h_tangent.reshape(-1, h_tangent.size(-1))
        e_flat = e_hidden.reshape(-1, e_hidden.size(-1))
        if h_flat.size(0) <= self.alignment_sample_size:
            return h_flat, e_flat

        index = torch.randperm(h_flat.size(0), device=h_flat.device)[
            : self.alignment_sample_size
        ]
        return h_flat[index], e_flat[index]

    def _compute_alignment_loss(
        self, h_tangent: torch.Tensor, e_hidden: torch.Tensor
    ) -> torch.Tensor | None:
        if (
            not self.use_alignment_loss
            or not self.training
            or self.align_h_proj is None
            or self.align_e_proj is None
        ):
            return None

        h_sample, e_sample = self._sample_alignment_features(h_tangent, e_hidden)
        h_proj = self.align_h_proj(h_sample)
        e_proj = self.align_e_proj(e_sample)
        return self.alignment_weight * self.alignment_loss_fn(h_proj, e_proj)

    def _compute_topology_mutual_loss(
        self, h_tangent: torch.Tensor, e_hidden: torch.Tensor
    ) -> torch.Tensor | None:
        if not self.use_topology_mutual_loss or not self.training:
            return None
        topology_loss = self.topology_mutual_loss_fn(h_tangent, e_hidden)
        return self.topology_mutual_weight * topology_loss

    def _compute_branch_scores(
        self, h_feature: torch.Tensor, e_feature: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.head_h is None or self.head_e is None:
            raise RuntimeError("Branch heads are required for DML or gated readout")
        score_h = self.head_h(torch.cat([h_feature, query], dim=-1)).squeeze(-1)
        score_e = self.head_e(torch.cat([e_feature, query], dim=-1)).squeeze(-1)
        return score_h, score_e

    def _compute_decision_mutual_loss(
        self,
        score_h: torch.Tensor,
        score_e: torch.Tensor,
        teacher_score: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not self.use_decision_mutual_loss or not self.training:
            return None
        decision_loss = self.decision_mutual_loss_fn(
            score_h,
            score_e,
            teacher_score=teacher_score,
            candidate_mask=candidate_mask,
            branch_kl_weight=self.decision_mutual_branch_kl_weight,
        )
        return self.decision_mutual_weight * decision_loss

    def _compute_branch_confidence(
        self, structure_score: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_e = ((structure_score - self.gate_tau_high) / (1 - self.gate_tau_high)).clamp(0, 1)
        weight_h = ((self.gate_tau_low - structure_score) / self.gate_tau_low).clamp(0, 1)
        return weight_h, weight_e

    def _require_structure_score(self, data: Any) -> torch.Tensor:
        if not hasattr(data, "structure_score"):
            raise ValueError(
                "Structure-gated readout requires graph data to contain structure_score"
            )
        return data.structure_score

    def _compute_specialization_loss(
        self,
        score_h: torch.Tensor,
        score_e: torch.Tensor,
        weight_h: torch.Tensor,
        weight_e: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.training or self.specialization_weight <= 0:
            return None
        pos_h = score_h[:, 0]
        pos_e = score_e[:, 0]
        pos_weight_h = weight_h[:, 0]
        pos_weight_e = weight_e[:, 0]
        specialization = self.specialization_loss_fn(
            pos_h,
            pos_e,
            pos_weight_h,
            pos_weight_e,
        )
        return self.specialization_weight * specialization

    def _set_auxiliary_state(
        self,
        alignment_loss: torch.Tensor | None,
        specialization_loss: torch.Tensor | None,
        stats: dict[str, torch.Tensor | float] | None = None,
        decision_mutual_loss: torch.Tensor | None = None,
        topology_mutual_loss: torch.Tensor | None = None,
    ) -> None:
        self._last_alignment_loss = alignment_loss
        self._last_decision_mutual_loss = decision_mutual_loss
        self._last_topology_mutual_loss = topology_mutual_loss
        self._last_specialization_loss = specialization_loss
        self._last_auxiliary_stats = {}
        if alignment_loss is not None:
            self._last_auxiliary_stats["alignment_loss"] = float(alignment_loss.detach().item())
        if decision_mutual_loss is not None:
            self._last_auxiliary_stats["decision_mutual_loss"] = float(
                decision_mutual_loss.detach().item()
            )
        if topology_mutual_loss is not None:
            self._last_auxiliary_stats["topology_mutual_loss"] = float(
                topology_mutual_loss.detach().item()
            )
            self._last_auxiliary_stats["topology_sample_size"] = float(
                self.topology_mutual_loss_fn.last_sample_size
            )
        if specialization_loss is not None:
            self._last_auxiliary_stats["specialization_loss"] = float(
                specialization_loss.detach().item()
            )
        if stats is None:
            return
        for key, value in stats.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    self._last_auxiliary_stats[key] = float(value.detach().item())
                else:
                    self._last_auxiliary_stats[key] = float(value.detach().mean().item())
            else:
                self._last_auxiliary_stats[key] = float(value)

    def get_auxiliary_loss(self) -> torch.Tensor | None:
        losses = [
            loss
            for loss in (
                self._last_alignment_loss,
                self._last_decision_mutual_loss,
                self._last_topology_mutual_loss,
                self._last_specialization_loss,
            )
            if loss is not None
        ]
        if not losses:
            return None
        total = losses[0]
        for loss in losses[1:]:
            total = total + loss
        return total

    def get_auxiliary_stats_dict(self) -> dict[str, float]:
        return dict(self._last_auxiliary_stats)


class RiemannianEntityNBFNet(_BaseRiemannianNBFNet):
    def bellmanford(
        self, data: Any, h_index: torch.Tensor, r_index: torch.Tensor
    ) -> dict[str, torch.Tensor | None]:
        batch_size = len(r_index)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        boundary = torch.zeros(
            batch_size, data.num_nodes, self.hidden_dim, device=h_index.device
        )
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        boundary = _clip_tangent_norm(boundary, self.boundary_max_tangent_norm)

        x_h = self._expmap_branch(self.manifold_h, boundary)
        boundary_h = x_h
        x_e = boundary if self.use_euclidean_branch else None
        boundary_e = boundary if self.use_euclidean_branch else None

        hiddens_h: list[torch.Tensor] = []
        hiddens_e: list[torch.Tensor] = []
        last_h_tangent = boundary
        last_e_hidden = boundary if self.use_euclidean_branch else None
        size = (data.num_nodes, data.num_nodes)
        euclidean_edge_weight = None
        if self.use_euclidean_branch:
            euclidean_edge_weight = torch.ones(data.edge_index.size(1), device=query.device)
            if self.euclidean_edge_weight_requires_grad:
                euclidean_edge_weight = euclidean_edge_weight.requires_grad_()
        for layer_idx, layer_h in enumerate(self.layers_h):
            if self.use_plain_euclidean_branch_one:
                output_h = layer_h(
                    x_h,
                    query,
                    boundary_h,
                    data.edge_index,
                    data.edge_type,
                    size,
                    euclidean_edge_weight,
                )
                if self.short_cut and output_h.shape == x_h.shape:
                    output_h = output_h + x_h
                x_h = output_h
                last_h_tangent = output_h
            else:
                output_h = layer_h(
                    x_h,
                    None,
                    query,
                    boundary_h,
                    None,
                    self.query,
                    data.edge_index,
                    data.edge_type,
                )
                x_h = output_h["x_h"]
                last_h_tangent = output_h["x_h_tangent"]
            if self.concat_hidden:
                hiddens_h.append(last_h_tangent)

            if self.use_euclidean_branch and self.layers_e is not None:
                layer_e = self.layers_e[layer_idx]
                output_e = layer_e(
                    x_e,
                    query,
                    boundary_e,
                    data.edge_index,
                    data.edge_type,
                    size,
                    euclidean_edge_weight,
                )
                if self.short_cut and output_e.shape == x_e.shape:
                    output_e = output_e + x_e
                x_e = output_e
                last_e_hidden = output_e
                if self.concat_hidden:
                    hiddens_e.append(output_e)

        query_nodes = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
        if self.concat_hidden:
            feature_parts = [*hiddens_h]
            if self.use_euclidean_branch:
                feature_parts.extend(hiddens_e)
            feature_parts.append(query_nodes)
        else:
            feature_parts = [last_h_tangent]
            if self.use_euclidean_branch and last_e_hidden is not None:
                feature_parts.append(last_e_hidden)
            feature_parts.append(query_nodes)
        node_feature = torch.cat(feature_parts, dim=-1)

        alignment_loss = None
        topology_mutual_loss = None
        if self.use_euclidean_branch and last_e_hidden is not None:
            alignment_loss = self._compute_alignment_loss(last_h_tangent, last_e_hidden)
            topology_mutual_loss = self._compute_topology_mutual_loss(
                last_h_tangent, last_e_hidden
            )
        return {
            "node_feature": node_feature,
            "x_h_tangent": last_h_tangent,
            "x_e_hidden": last_e_hidden,
            "query": query,
            "alignment_loss": alignment_loss,
            "topology_mutual_loss": topology_mutual_loss,
        }

    def forward(
        self,
        data: Any,
        relation_representations: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h_index, t_index, r_index = batch.unbind(-1)
        self.query = relation_representations
        self._bind_relations(relation_representations)

        shape = h_index.shape
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index, num_direct_rel=data.num_relations // 2
        )
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])
        alignment_loss = output["alignment_loss"]
        topology_mutual_loss = output["topology_mutual_loss"]
        feature = output["node_feature"]
        gather_index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        gathered_feature = feature.gather(1, gather_index)
        score_fuse = self.mlp(gathered_feature).squeeze(-1)
        needs_branch_scores = self.use_structure_gated_readout or (
            self.training and self.use_decision_mutual_loss
        )

        if not self.use_euclidean_branch or not needs_branch_scores:
            self._set_auxiliary_state(
                alignment_loss,
                None,
                decision_mutual_loss=None,
                topology_mutual_loss=topology_mutual_loss,
            )
            return score_fuse.view(shape)

        x_h_tangent = output["x_h_tangent"]
        x_e_hidden = output["x_e_hidden"]
        assert x_e_hidden is not None
        gather_branch_index = t_index.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        h_t = x_h_tangent.gather(1, gather_branch_index)
        e_t = x_e_hidden.gather(1, gather_branch_index)
        query = output["query"]
        q_t = query.unsqueeze(1).expand(-1, t_index.size(1), -1)

        score_h, score_e = self._compute_branch_scores(h_t, e_t, q_t)
        teacher_score = score_fuse if self.decision_mutual_teacher == "fuse" else None
        decision_mutual_loss = self._compute_decision_mutual_loss(
            score_h,
            score_e,
            teacher_score=teacher_score,
        )
        score = score_fuse
        specialization_loss = None
        stats = {
            "branch_score_gap": (score_h - score_e).mean(),
        }

        if self.use_structure_gated_readout:
            structure_score = self._require_structure_score(data)[t_index]
            weight_h, weight_e = self._compute_branch_confidence(structure_score)
            score = score_fuse + self.gate_beta * (
                weight_h * score_h + weight_e * score_e
            )
            specialization_loss = self._compute_specialization_loss(
                score_h, score_e, weight_h, weight_e
            )
            stats.update(
                {
                    "pos_score_gap": (score_h[:, 0] - score_e[:, 0]).mean(),
                    "high_conf_lorentz_positive_rate": (weight_h[:, 0] > 0)
                    .float()
                    .mean(),
                    "high_conf_euclidean_positive_rate": (weight_e[:, 0] > 0)
                    .float()
                    .mean(),
                    "positive_residual_confidence": (
                        weight_h[:, 0] + weight_e[:, 0]
                    ).mean(),
                }
            )

        self._set_auxiliary_state(
            alignment_loss,
            specialization_loss,
            stats,
            decision_mutual_loss=decision_mutual_loss,
            topology_mutual_loss=topology_mutual_loss,
        )
        return score.view(shape)


class RiemannianQueryNBFNet(RiemannianEntityNBFNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_node_tangent: torch.Tensor | None = None
        self._last_question_entity_node_repr: torch.Tensor | None = None
        self._last_branch_scores: dict[str, torch.Tensor] | None = None

    def bellmanford(
        self, data: Any, node_features: torch.Tensor, query: torch.Tensor
    ) -> dict[str, torch.Tensor | None]:
        self._last_node_tangent = None
        self._last_question_entity_node_repr = None
        self._last_branch_scores = None
        node_features = _clip_tangent_norm(
            node_features, self.boundary_max_tangent_norm
        )
        x_h = self._expmap_branch(self.manifold_h, node_features)
        boundary_h = x_h
        x_e = node_features if self.use_euclidean_branch else None
        boundary_e = node_features if self.use_euclidean_branch else None

        hiddens_h: list[torch.Tensor] = []
        hiddens_e: list[torch.Tensor] = []
        last_h_tangent = node_features
        last_e_hidden = node_features if self.use_euclidean_branch else None
        size = (data.num_nodes, data.num_nodes)
        euclidean_edge_weight = None
        if self.use_euclidean_branch:
            euclidean_edge_weight = torch.ones(data.edge_index.size(1), device=query.device)
            if self.euclidean_edge_weight_requires_grad:
                euclidean_edge_weight = euclidean_edge_weight.requires_grad_()
        for layer_idx, layer_h in enumerate(self.layers_h):
            if self.use_plain_euclidean_branch_one:
                output_h = layer_h(
                    x_h,
                    query,
                    boundary_h,
                    data.edge_index,
                    data.edge_type,
                    size,
                    euclidean_edge_weight,
                )
                if self.short_cut and output_h.shape == x_h.shape:
                    output_h = output_h + x_h
                x_h = output_h
                last_h_tangent = output_h
            else:
                output_h = layer_h(
                    x_h,
                    None,
                    query,
                    boundary_h,
                    None,
                    self.query,
                    data.edge_index,
                    data.edge_type,
                )
                x_h = output_h["x_h"]
                last_h_tangent = output_h["x_h_tangent"]
            if self.concat_hidden:
                hiddens_h.append(last_h_tangent)

            if self.use_euclidean_branch and self.layers_e is not None:
                layer_e = self.layers_e[layer_idx]
                output_e = layer_e(
                    x_e,
                    query,
                    boundary_e,
                    data.edge_index,
                    data.edge_type,
                    size,
                    euclidean_edge_weight,
                )
                if self.short_cut and output_e.shape == x_e.shape:
                    output_e = output_e + x_e
                x_e = output_e
                last_e_hidden = output_e
                if self.concat_hidden:
                    hiddens_e.append(output_e)

        query_nodes = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
        if self.concat_hidden:
            feature_parts = [*hiddens_h]
            if self.use_euclidean_branch:
                feature_parts.extend(hiddens_e)
            feature_parts.append(query_nodes)
        else:
            feature_parts = [last_h_tangent]
            if self.use_euclidean_branch and last_e_hidden is not None:
                feature_parts.append(last_e_hidden)
            feature_parts.append(query_nodes)
        node_feature = torch.cat(feature_parts, dim=-1)

        alignment_loss = None
        topology_mutual_loss = None
        if self.use_euclidean_branch and last_e_hidden is not None:
            alignment_loss = self._compute_alignment_loss(last_h_tangent, last_e_hidden)
            topology_mutual_loss = self._compute_topology_mutual_loss(
                last_h_tangent, last_e_hidden
            )
            self._last_question_entity_node_repr = torch.cat(
                [last_h_tangent, last_e_hidden], dim=-1
            )
        else:
            self._last_node_tangent = last_h_tangent
            self._last_question_entity_node_repr = last_h_tangent
        return {
            "node_feature": node_feature,
            "x_h_tangent": last_h_tangent,
            "x_e_hidden": last_e_hidden,
            "alignment_loss": alignment_loss,
            "topology_mutual_loss": topology_mutual_loss,
        }

    def forward(
        self,
        data: Any,
        node_features: torch.Tensor,
        relation_representations: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        self.query = relation_representations
        self._bind_relations(relation_representations)
        output = self.bellmanford(data, node_features, query)
        alignment_loss = output["alignment_loss"]
        topology_mutual_loss = output["topology_mutual_loss"]
        score_fuse = self.mlp(output["node_feature"]).squeeze(-1)
        needs_branch_scores = (
            self.use_structure_gated_readout
            or self.needs_branch_supervision
            or (self.training and self.use_decision_mutual_loss)
        )
        if not self.use_euclidean_branch or not needs_branch_scores:
            self._set_auxiliary_state(
                alignment_loss,
                None,
                decision_mutual_loss=None,
                topology_mutual_loss=topology_mutual_loss,
            )
            return score_fuse

        x_h_tangent = output["x_h_tangent"]
        x_e_hidden = output["x_e_hidden"]
        assert x_e_hidden is not None
        query_nodes = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
        score_h, score_e = self._compute_branch_scores(
            x_h_tangent, x_e_hidden, query_nodes
        )
        self._last_branch_scores = {
            "fuse": score_fuse,
            "h": score_h,
            "e": score_e,
        }
        teacher_score = score_fuse if self.decision_mutual_teacher == "fuse" else None
        decision_mutual_loss = None
        if self.decision_mutual_candidate_mode == "all":
            decision_mutual_loss = self._compute_decision_mutual_loss(
                score_h,
                score_e,
                teacher_score=teacher_score,
            )
        score = score_fuse
        stats = {
            "branch_score_gap": (score_h - score_e).mean(),
        }
        if self.use_structure_gated_readout:
            structure_score = self._require_structure_score(data).unsqueeze(0).expand(
                query.size(0), -1
            )
            weight_h, weight_e = self._compute_branch_confidence(structure_score)
            score = score_fuse + self.gate_beta * (
                weight_h * score_h + weight_e * score_e
            )
            stats.update(
                {
                    "high_conf_lorentz_positive_rate": (weight_h > 0)
                    .float()
                    .mean(),
                    "high_conf_euclidean_positive_rate": (weight_e > 0)
                    .float()
                    .mean(),
                    "positive_residual_confidence": (weight_h + weight_e).mean(),
                }
            )
        self._set_auxiliary_state(
            alignment_loss,
            None,
            stats,
            decision_mutual_loss=decision_mutual_loss,
            topology_mutual_loss=topology_mutual_loss,
        )
        return score

    def get_last_node_tangent(self) -> torch.Tensor | None:
        return self._last_node_tangent

    def get_question_entity_node_repr(self) -> torch.Tensor | None:
        return self._last_question_entity_node_repr

    def get_last_branch_scores(self) -> dict[str, torch.Tensor] | None:
        if self._last_branch_scores is None:
            return None
        return dict(self._last_branch_scores)

    def visualize(self, *args: Any, **kwargs: Any) -> dict:
        raise NotImplementedError(
            "Visualization is not implemented for RiemannianQueryNBFNet"
        )
