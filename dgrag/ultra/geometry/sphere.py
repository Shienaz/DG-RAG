from __future__ import annotations

import geoopt
import torch

from dgrag.ultra.variadic import native_scatter

EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


def _sin_div(x: torch.Tensor) -> torch.Tensor:
    x_sq = x * x
    # Use a low-order series near zero to avoid 0/0 in backward.
    series = 1 - x_sq / 6 + (x_sq * x_sq) / 120
    denom = torch.where(x.abs() > 1e-8, x, torch.ones_like(x))
    ratio = torch.sin(x) / denom
    return torch.where(x.abs() > 1e-4, ratio, series)


class Sphere(geoopt.Sphere):
    def __init__(self, learnable: bool = False) -> None:
        super().__init__()
        self.k = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=learnable)

    def origin(
        self,
        size: tuple[int, ...] | torch.Size,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        pole = torch.zeros(size, dtype=dtype, device=device)
        pole[..., 0] = -1
        return pole

    def cinner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x @ y.transpose(-1, -2)

    def inner(
        self,
        x: torch.Tensor | None,
        u: torch.Tensor,
        v: torch.Tensor | None = None,
        *,
        keepdim: bool = False,
    ) -> torch.Tensor:
        del x
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def expmap0(self, u: torch.Tensor, dim: int = -1) -> torch.Tensor:
        del dim
        origin = self.origin(u.shape, dtype=u.dtype, device=u.device)
        return self.expmap(origin, u)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * _sin_div(norm_u)
        retr = self.projx(x + u)
        cond = norm_u > EPS[norm_u.dtype]
        return torch.where(cond, exp, retr)

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        x = self.origin(y.shape, dtype=y.dtype, device=y.device)
        u = self.proju(x, y - x)
        dist = self.dist(x, y, keepdim=True)
        cond = dist > EPS[y.dtype]
        return torch.where(
            cond,
            u * dist / u.norm(dim=-1, keepdim=True).clamp_min(EPS[y.dtype]),
            u,
        )

    def proju0(self, u: torch.Tensor) -> torch.Tensor:
        x = self.origin(u.shape, dtype=u.dtype, device=u.device)
        u = u - (x * u).sum(dim=-1, keepdim=True) * x
        return self._project_on_subspace(u)

    def norm(
        self,
        u: torch.Tensor,
        x: torch.Tensor | None = None,
        *,
        keepdim: bool = False,
    ) -> torch.Tensor:
        del x
        return torch.norm(u, dim=-1, keepdim=keepdim)

    def random_normal(
        self,
        *size: int,
        mean: float = 0,
        std: float = 1,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> geoopt.ManifoldTensor:
        tens = torch.randn(*size, device=device, dtype=dtype) * std + mean
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)

    def transp0back(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        origin = self.origin(x.shape, dtype=x.dtype, device=x.device)
        return self.transp(x, origin, u)

    def Frechet_mean(
        self,
        x: torch.Tensor,
        weights: torch.Tensor | None = None,
        dim: int = 0,
        keepdim: bool = False,
        sum_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if weights is None:
            z = (
                torch.sum(x, dim=dim, keepdim=keepdim)
                if sum_idx is None
                else native_scatter(x, sum_idx, dim=dim, reduce="sum")
            )
        else:
            z = (
                torch.sum(x * weights, dim=dim, keepdim=keepdim)
                if sum_idx is None
                else native_scatter(x * weights, sum_idx, dim=dim, reduce="sum")
            )
        return self.normalize_midpoint(z)

    def normalize_midpoint(self, z: torch.Tensor) -> torch.Tensor:
        denom = self.inner(None, z, keepdim=True).abs().clamp_min(1e-8).sqrt()
        return (1.0 / torch.sqrt(self.k).to(device=z.device, dtype=z.dtype)) * z / denom
