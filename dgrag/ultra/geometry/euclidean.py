from __future__ import annotations

import torch
from torch import nn


class Euclidean(nn.Module):
    """Flat geometry for the parameter-matched Dual-Euclidean control.

    All coordinates are spatial. Maps at the origin and tangent projection are
    identities. The attention layer normalizes by the sum of message weights.
    """

    def proju0(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def expmap0(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def logmap0(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def normalize_midpoint(self, value: torch.Tensor) -> torch.Tensor:
        return value
