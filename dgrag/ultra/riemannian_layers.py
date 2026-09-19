from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F  # noqa:N812
from torch.distributed.nn.functional import all_reduce as differentiable_all_reduce

from dgrag.ultra.geometry import Euclidean, Lorentz, Sphere
from dgrag.ultra.variadic import native_scatter, native_scatter_softmax


def _broadcast_query(query: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return query.unsqueeze(1).expand(-1, num_nodes, -1)


def _tangent_clip_scale(
    tangent: torch.Tensor, max_norm: float | None
) -> torch.Tensor | None:
    """Return the per-vector rescaling factor applied by ``_clip_tangent_norm``."""
    if max_norm is None:
        return None

    norm = tangent.norm(dim=-1, keepdim=True)
    return torch.clamp(max_norm / norm.clamp_min(1e-8), max=1.0)


def _clip_tangent_norm(
    tangent: torch.Tensor, max_norm: float | None
) -> torch.Tensor:
    scale = _tangent_clip_scale(tangent, max_norm)
    if scale is None:
        return tangent

    return tangent * scale


class RiemannianRelationalConv(nn.Module):
    """Relation-aware tangent-space message passing with dual manifold branches."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        query_input_dim: int,
        branch_geometries: Sequence[str] | None = None,
        message_func: str = "distmult",
        aggregate_func: str = "attention_frechet",
        layer_norm: bool = False,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = "relu",
        residual_in_tangent_space: bool = True,
        learnable_curvature: bool = False,
        boundary_max_tangent_norm: float | None = 10.0,
        message_max_tangent_norm: float | None = 10.0,
        update_max_tangent_norm: float | None = 10.0,
        use_edge_parallel: bool = False,
        edge_parallel_shard_strategy: str = "contiguous",
    ) -> None:
        super().__init__()
        if message_func != "distmult":
            raise ValueError("RiemannianRelationalConv currently supports only distmult")
        if aggregate_func != "attention_frechet":
            raise ValueError(
                "RiemannianRelationalConv currently supports only attention_frechet"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.residual_in_tangent_space = residual_in_tangent_space
        self.branch_geometries = list(branch_geometries or ["lorentz", "sphere"])
        if self.branch_geometries not in (["lorentz"], ["lorentz", "sphere"]):
            raise ValueError(
                "RiemannianRelationalConv supports only ['lorentz'] or ['lorentz', 'sphere']"
            )
        self.use_sphere_branch = "sphere" in self.branch_geometries
        self.num_branches = len(self.branch_geometries)
        self.boundary_max_tangent_norm = boundary_max_tangent_norm
        self.message_max_tangent_norm = message_max_tangent_norm
        self.update_max_tangent_norm = update_max_tangent_norm
        if edge_parallel_shard_strategy != "contiguous":
            raise ValueError("edge_parallel_shard_strategy currently supports only 'contiguous'")
        self.use_edge_parallel = use_edge_parallel
        self.edge_parallel_shard_strategy = edge_parallel_shard_strategy

        self.manifold_h = Lorentz(learnable=learnable_curvature)
        self.manifold_s = (
            Sphere(learnable=learnable_curvature) if self.use_sphere_branch else None
        )

        self.input_h_proj = nn.Linear(input_dim, output_dim)
        self.boundary_h_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.relation_h_proj = nn.Linear(query_input_dim, output_dim, bias=False)
        self.query_proj = nn.Linear(query_input_dim, output_dim, bias=False)
        if self.use_sphere_branch:
            self.input_s_proj = nn.Linear(input_dim, output_dim)
            self.boundary_s_proj = nn.Linear(input_dim, output_dim, bias=False)
            self.relation_s_proj = nn.Linear(query_input_dim, output_dim, bias=False)
        else:
            self.input_s_proj = None
            self.boundary_s_proj = None
            self.relation_s_proj = None

        attn_dim = output_dim * 3
        self.attn_h = nn.Linear(attn_dim, 1)
        self.attn_s = nn.Linear(attn_dim, 1) if self.use_sphere_branch else None

        update_input_dim = output_dim * (3 * self.num_branches + 1)
        self.update_mlp = nn.Sequential(
            nn.Linear(update_input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim * self.num_branches),
        )

        if layer_norm:
            self.layer_norm_h = nn.LayerNorm(output_dim)
            self.layer_norm_s = (
                nn.LayerNorm(output_dim) if self.use_sphere_branch else None
            )
        else:
            self.layer_norm_h = None
            self.layer_norm_s = None

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def _expmap_branch(self, manifold: Lorentz | Sphere | Euclidean, tangent: torch.Tensor) -> torch.Tensor:
        flat = tangent.reshape(-1, tangent.size(-1))
        flat = manifold.proju0(flat)
        return manifold.expmap0(flat).reshape_as(tangent)

    def _logmap_branch(self, manifold: Lorentz | Sphere | Euclidean, points: torch.Tensor) -> torch.Tensor:
        flat = points.reshape(-1, points.size(-1))
        return manifold.logmap0(flat).reshape_as(points)

    def _edge_parallel_enabled(self) -> bool:
        return (
            self.use_edge_parallel
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    def _edge_shard(
        self, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        num_edges = edge_type.numel()
        start = (num_edges * rank) // world_size
        end = (num_edges * (rank + 1)) // world_size
        return edge_index[:, start:end], edge_type[start:end]

    def _distributed_edge_softmax(
        self, score: torch.Tensor, dst: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        # Empty shards must still participate in both softmax collectives.
        index = dst.unsqueeze(0).expand_as(score)
        local_max = score.new_full(
            (score.size(0), num_nodes), torch.finfo(score.dtype).min
        )
        local_max.scatter_reduce_(
            1, index, score, reduce="amax", include_self=True
        )
        with torch.no_grad():
            global_max = local_max.clone()
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX)

        stable_score = score - global_max.gather(1, index)
        exp_score = torch.exp(stable_score)
        local_denominator = score.new_zeros((score.size(0), num_nodes))
        local_denominator.scatter_add_(1, index, exp_score)
        global_denominator = differentiable_all_reduce(
            local_denominator, op=dist.ReduceOp.SUM
        )
        return exp_score / global_denominator.gather(1, index).clamp_min(1e-12)

    def _attention_terms(
        self,
        state: torch.Tensor,
        relation_repr: torch.Tensor,
        query: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-edge operands of the attention score, shared by every reduce path."""
        src, _ = edge_index
        state_src = state[:, src, :]
        rel_edge = relation_repr[:, edge_type, :]
        query_edge = query.unsqueeze(1).expand(-1, edge_type.numel(), -1)
        return state_src, rel_edge, query_edge

    def _attention_scores(
        self,
        attn_linear: nn.Linear,
        state_src: torch.Tensor,
        rel_edge: torch.Tensor,
        query_edge: torch.Tensor,
    ) -> torch.Tensor:
        attention_input = torch.cat([state_src, rel_edge, query_edge], dim=-1)
        return attn_linear(attention_input).squeeze(-1)

    def _attention_weights(
        self,
        attn_linear: nn.Linear,
        state_src: torch.Tensor,
        rel_edge: torch.Tensor,
        query_edge: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        attn_score = self._attention_scores(
            attn_linear, state_src, rel_edge, query_edge
        )
        attn_index = dst.unsqueeze(0).expand_as(attn_score)
        return native_scatter_softmax(
            attn_score, attn_index, dim=1, dim_size=num_nodes
        )

    def _prepare_messages(
        self,
        state: torch.Tensor,
        relation_repr: torch.Tensor,
        query: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        attn_linear: nn.Linear,
        manifold: Lorentz | Sphere | Euclidean,
        boundary_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._edge_parallel_enabled():
            return self._prepare_messages_edge_parallel(
                state,
                relation_repr,
                query,
                edge_index,
                edge_type,
                attn_linear,
                manifold,
                boundary_points,
            )

        _, dst = edge_index
        state_src, rel_edge, query_edge = self._attention_terms(
            state, relation_repr, query, edge_index, edge_type
        )

        message_tangent = state_src * rel_edge
        message_tangent = _clip_tangent_norm(
            message_tangent, self.message_max_tangent_norm
        )
        attn_weight = self._attention_weights(
            attn_linear,
            state_src,
            rel_edge,
            query_edge,
            dst,
            boundary_points.size(1),
        )

        message_points = self._expmap_branch(manifold, message_tangent)
        weighted_points = message_points * attn_weight.unsqueeze(-1)

        boundary_weight = torch.ones(
            weighted_points.size(0),
            boundary_points.size(1),
            device=weighted_points.device,
            dtype=weighted_points.dtype,
        )
        values = torch.cat(
            [weighted_points, boundary_points * boundary_weight.unsqueeze(-1)], dim=1
        )
        index = torch.cat(
            [dst, torch.arange(boundary_points.size(1), device=dst.device)], dim=0
        )
        midpoint_sum = native_scatter(values, index, dim=1, dim_size=boundary_points.size(1), reduce="sum")
        midpoint = manifold.normalize_midpoint(midpoint_sum)
        return midpoint, attn_weight

    def _prepare_messages_edge_parallel(
        self,
        state: torch.Tensor,
        relation_repr: torch.Tensor,
        query: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        attn_linear: nn.Linear,
        manifold: Lorentz | Sphere | Euclidean,
        boundary_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_edge_index, local_edge_type = self._edge_shard(edge_index, edge_type)
        _, dst = local_edge_index
        num_nodes = boundary_points.size(1)

        state_src, rel_edge, query_edge = self._attention_terms(
            state, relation_repr, query, local_edge_index, local_edge_type
        )

        message_tangent = state_src * rel_edge
        message_tangent = _clip_tangent_norm(
            message_tangent, self.message_max_tangent_norm
        )
        attn_score = self._attention_scores(
            attn_linear, state_src, rel_edge, query_edge
        )
        attn_weight = self._distributed_edge_softmax(attn_score, dst, num_nodes)

        message_points = self._expmap_branch(manifold, message_tangent)
        weighted_points = message_points * attn_weight.unsqueeze(-1)
        local_sum = native_scatter(
            weighted_points,
            dst,
            dim=1,
            dim_size=num_nodes,
            reduce="sum",
        )
        global_edge_sum = differentiable_all_reduce(
            local_sum, op=dist.ReduceOp.SUM
        )
        midpoint = manifold.normalize_midpoint(global_edge_sum + boundary_points)
        return midpoint, attn_weight

    def forward(
        self,
        input_h: torch.Tensor,
        input_s: torch.Tensor | None,
        query: torch.Tensor,
        boundary_h: torch.Tensor,
        boundary_s: torch.Tensor | None,
        relation_representations: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_nodes = input_h.size(1)

        input_h_tangent = self.input_h_proj(self._logmap_branch(self.manifold_h, input_h))
        boundary_h_tangent = self.boundary_h_proj(
            self._logmap_branch(self.manifold_h, boundary_h)
        )
        boundary_h_tangent = _clip_tangent_norm(
            boundary_h_tangent, self.boundary_max_tangent_norm
        )
        query_tangent = self.query_proj(query)
        relation_h_tangent = self.relation_h_proj(relation_representations)

        boundary_h_points = self._expmap_branch(self.manifold_h, boundary_h_tangent)

        aggregated_h_points, _ = self._prepare_messages(
            input_h_tangent,
            relation_h_tangent,
            query_tangent,
            edge_index,
            edge_type,
            self.attn_h,
            self.manifold_h,
            boundary_h_points,
        )
        aggregated_h_tangent = self._logmap_branch(self.manifold_h, aggregated_h_points)
        update_parts = [input_h_tangent]

        if self.use_sphere_branch:
            if (
                input_s is None
                or boundary_s is None
                or self.input_s_proj is None
                or self.boundary_s_proj is None
                or self.relation_s_proj is None
                or self.attn_s is None
                or self.manifold_s is None
            ):
                raise ValueError("Sphere branch inputs and modules must be provided")
            input_s_tangent = self.input_s_proj(
                self._logmap_branch(self.manifold_s, input_s)
            )
            boundary_s_tangent = self.boundary_s_proj(
                self._logmap_branch(self.manifold_s, boundary_s)
            )
            boundary_s_tangent = _clip_tangent_norm(
                boundary_s_tangent, self.boundary_max_tangent_norm
            )
            relation_s_tangent = self.relation_s_proj(relation_representations)
            boundary_s_points = self._expmap_branch(self.manifold_s, boundary_s_tangent)
            aggregated_s_points, _ = self._prepare_messages(
                input_s_tangent,
                relation_s_tangent,
                query_tangent,
                edge_index,
                edge_type,
                self.attn_s,
                self.manifold_s,
                boundary_s_points,
            )
            aggregated_s_tangent = self._logmap_branch(
                self.manifold_s, aggregated_s_points
            )
            update_parts.extend(
                [input_s_tangent, aggregated_h_tangent, aggregated_s_tangent]
            )
            update_parts.extend([boundary_h_tangent, boundary_s_tangent])
        else:
            input_s_tangent = None
            boundary_s_tangent = None
            aggregated_s_tangent = None
            update_parts.extend([aggregated_h_tangent, boundary_h_tangent])

        query_nodes = _broadcast_query(query_tangent, num_nodes)
        update_parts.append(query_nodes)
        update = self.update_mlp(torch.cat(update_parts, dim=-1))

        if self.use_sphere_branch:
            update_h_tangent, update_s_tangent = update.chunk(2, dim=-1)
        else:
            update_h_tangent = update
            update_s_tangent = None

        if self.residual_in_tangent_space and input_h_tangent.shape == update_h_tangent.shape:
            update_h_tangent = update_h_tangent + input_h_tangent
            if update_s_tangent is not None and input_s_tangent is not None:
                update_s_tangent = update_s_tangent + input_s_tangent

        if self.layer_norm_h is not None:
            update_h_tangent = self.layer_norm_h(update_h_tangent)
        if self.layer_norm_s is not None and update_s_tangent is not None:
            update_s_tangent = self.layer_norm_s(update_s_tangent)

        if self.activation is not None:
            update_h_tangent = self.activation(update_h_tangent)
            if update_s_tangent is not None:
                update_s_tangent = self.activation(update_s_tangent)

        update_h_tangent = _clip_tangent_norm(
            update_h_tangent, self.update_max_tangent_norm
        )
        output_h = self._expmap_branch(self.manifold_h, update_h_tangent)
        output = {
            "x_h": output_h,
            "x_h_tangent": update_h_tangent,
        }
        if update_s_tangent is not None and self.manifold_s is not None:
            update_s_tangent = _clip_tangent_norm(
                update_s_tangent, self.update_max_tangent_norm
            )
            output["x_s"] = self._expmap_branch(self.manifold_s, update_s_tangent)
            output["x_s_tangent"] = update_s_tangent
        else:
            output["x_s"] = None
            output["x_s_tangent"] = None
        return output


class EuclideanAttentionRelationalConv(RiemannianRelationalConv):
    """Flat counterpart retaining the Lorentz branch's learned architecture.

    Attention, projections, clipping, update MLP, and residuals are unchanged.
    Maps become identities; aggregation becomes a Euclidean weighted mean.
    The inherited *_h names denote checkpoint slots, not Lorentz operations.

    Because the flat maps are identities, the weighted mean

        out[d] = sum_{e: dst=d} attn[e] * clip(state[src[e]] * rel[type[e]])

    is a plain relational sparse sum whose per-edge scalar is
    ``attn[e] * clip_scale[e]``. It therefore also runs on the fused ``rspmm``
    kernel (``use_rspmm_aggregation``), which the Lorentz branch cannot use: the
    Frechet midpoint needs the per-edge message in the manifold.
    """

    def __init__(
        self,
        *args: Any,
        branch_geometries: Sequence[str] | None = None,
        use_rspmm_aggregation: bool = True,
        **kwargs: Any,
    ) -> None:
        if branch_geometries is not None and list(branch_geometries) != ["euclidean"]:
            raise ValueError("Euclidean attention requires branch_geometries=['euclidean']")
        if kwargs.get("learnable_curvature", False):
            raise ValueError("Dual-Euclidean requires learnable_curvature=false")
        super().__init__(*args, branch_geometries=["lorentz"], **kwargs)
        self.branch_geometries = ["euclidean"]
        self.manifold_h = Euclidean()
        self.use_rspmm_aggregation = use_rspmm_aggregation

    def _rspmm_aggregation_enabled(
        self, state: torch.Tensor, edge_type: torch.Tensor
    ) -> bool:
        """Keep the fused kernel opt-in per device; CPU keeps the native path."""
        if not self.use_rspmm_aggregation or not state.is_cuda:
            return False
        if edge_type.numel() == 0:
            return False
        return not self._edge_parallel_enabled()

    def _rspmm_weighted_sum(
        self,
        state: torch.Tensor,
        relation_repr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor,
        boundary_points: torch.Tensor,
    ) -> torch.Tensor:
        """``sum_e edge_weight[e] * (relation[type[e]] * state[src[e]])`` + boundary.

        ``generalized_rspmm`` reduces into ``edge_index[0]`` while reading
        ``input`` at ``edge_index[1]``; this layer's attention path reads the
        source at ``edge_index[0]`` and reduces into ``edge_index[1]``. The edge
        list is transposed for the call so both paths traverse identically.

        The kernel takes one scalar per edge and folds the batch into the feature
        dimension, while attention weights are per sample, so one call is issued
        per batch element.
        """
        from dgrag.ultra.rspmm import generalized_rspmm

        rspmm_edge_index = edge_index.flip(0)
        per_sample = [
            generalized_rspmm(
                rspmm_edge_index,
                edge_type,
                edge_weight[index],
                relation_repr[index],
                state[index],
                sum="add",
                mul="mul",
            )
            for index in range(state.size(0))
        ]
        return torch.stack(per_sample, dim=0) + boundary_points

    def _prepare_messages_rspmm(
        self,
        state: torch.Tensor,
        relation_repr: torch.Tensor,
        query: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        attn_linear: nn.Linear,
        boundary_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, dst = edge_index
        state_src, rel_edge, query_edge = self._attention_terms(
            state, relation_repr, query, edge_index, edge_type
        )
        message_tangent = state_src * rel_edge
        clip_scale = _tangent_clip_scale(
            message_tangent, self.message_max_tangent_norm
        )
        attn_weight = self._attention_weights(
            attn_linear,
            state_src,
            rel_edge,
            query_edge,
            dst,
            boundary_points.size(1),
        )
        edge_weight = attn_weight if clip_scale is None else attn_weight * clip_scale.squeeze(-1)
        weighted_sum = self._rspmm_weighted_sum(
            state,
            relation_repr,
            edge_index,
            edge_type,
            edge_weight,
            boundary_points,
        )
        return weighted_sum, attn_weight

    def _prepare_messages(
        self,
        state: torch.Tensor,
        relation_repr: torch.Tensor,
        query: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        attn_linear: nn.Linear,
        manifold: Euclidean,
        boundary_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rspmm_aggregation_enabled(state, edge_type):
            weighted_sum, attention = self._prepare_messages_rspmm(
                state, relation_repr, query, edge_index, edge_type,
                attn_linear, boundary_points,
            )
        else:
            weighted_sum, attention = super()._prepare_messages(
                state, relation_repr, query, edge_index, edge_type,
                attn_linear, manifold, boundary_points,
            )
        # Incoming attention sums to one; the boundary has weight one. Use the
        # global edge list also in edge-parallel mode. Isolated nodes have only
        # the boundary and must not be divided by two.
        has_incoming = weighted_sum.new_zeros(boundary_points.size(1))
        has_incoming[edge_index[1]] = 1
        return weighted_sum / (1 + has_incoming)[None, :, None], attention
