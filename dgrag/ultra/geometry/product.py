from __future__ import annotations

from typing import Tuple, Union

import geoopt
import torch


class ProductSpace(geoopt.ProductManifold):
    def __init__(
        self,
        *manifolds_with_shape: Tuple[geoopt.Manifold, Union[Tuple[int, ...], int]],
    ) -> None:
        super().__init__(*manifolds_with_shape)

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        target_batch_dim = x.dim() - 1
        outputs = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            logmapped = manifold.logmap0(point)
            outputs.append(logmapped.reshape((*logmapped.shape[:target_batch_dim], -1)))
        return torch.cat(outputs, dim=-1)

    def proju0(self, u: torch.Tensor) -> torch.Tensor:
        target_batch_dim = u.dim() - 1
        outputs = []
        for i, manifold in enumerate(self.manifolds):
            tangent = self.take_submanifold_value(u, i)
            proj = manifold.proju0(tangent)
            outputs.append(proj.reshape((*proj.shape[:target_batch_dim], -1)))
        return torch.cat(outputs, dim=-1)

    def Frechet_mean(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None = None,
        dim: int = 0,
        keepdim: bool = False,
        sum_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_batch_dim = x.dim() - 1
        outputs = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            midpoint = manifold.Frechet_mean(point, weights, dim, keepdim, sum_idx)
            outputs.append(midpoint.reshape((*midpoint.shape[:target_batch_dim], -1)))
        return torch.cat(outputs, dim=-1)
