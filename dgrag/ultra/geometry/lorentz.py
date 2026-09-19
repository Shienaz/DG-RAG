from __future__ import annotations

import geoopt
import torch

from dgrag.ultra.variadic import native_scatter

EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


def _sinh_div(x: torch.Tensor) -> torch.Tensor:
    x_sq = x * x
    # Use a low-order series near zero to avoid 0/0 in backward.
    series = 1 + x_sq / 6 + (x_sq * x_sq) / 120
    denom = torch.where(x.abs() > 1e-8, x, torch.ones_like(x))
    ratio = torch.sinh(x) / denom
    return torch.where(x.abs() > 1e-4, ratio, series)


class Lorentz(geoopt.Lorentz):
    def __init__(self, k: float = 1.0, learnable: bool = False) -> None:
        super().__init__(k=k, learnable=learnable)

    def origin(
        self,
        size: tuple[int, ...] | torch.Size,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        origin = torch.zeros(size, dtype=dtype, device=device)
        origin[..., 0] = torch.sqrt(self.k).to(device=device, dtype=dtype)
        return origin

    def cinner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[..., 0].mul_(-1)
        return x @ y.transpose(-1, -2)

    def expmap0(self, u: torch.Tensor, dim: int = -1) -> torch.Tensor:
        origin = self.origin(u.shape, dtype=u.dtype, device=u.device)
        return self.expmap(origin, u, dim=dim)

    def expmap(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        *,
        norm_tan: bool = False,
        project: bool = False,
        dim: int = -1,
    ) -> torch.Tensor:
        del norm_tan, project
        denom = torch.sqrt(self.k).to(device=u.device, dtype=u.dtype)
        norm_u = self.norm(u, keepdim=True, dim=dim).clamp_min(1e-8)
        return torch.cosh(norm_u / denom) * x + _sinh_div(norm_u / denom) * u

    def logmap0(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        d = x.size(dim) - 1
        y = x.narrow(dim, 1, d).reshape(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True).clamp_min(1e-8)
        sqrt_k = torch.sqrt(self.k).to(device=x.device, dtype=x.dtype)
        theta = torch.clamp(
            x.narrow(dim, 0, 1).reshape(-1, 1) / sqrt_k,
            min=1.0 + EPS[x.dtype],
        )
        result = torch.zeros_like(x).reshape(-1, x.size(dim))
        result[:, 1:] = sqrt_k * torch.acosh(theta) * y / y_norm
        return result.reshape_as(x)

    def proju0(self, v: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
        origin = self.origin(v.shape, dtype=v.dtype, device=v.device)
        return self.proju(origin, v, dim=dim)

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
