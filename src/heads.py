"""Projection head MLP for mapping frozen LLM vectors onto the unit sphere.

Remaps R^{in_dim} -> R^{out_dim}:
  Linear -> LayerNorm -> GELU -> Linear -> L2-normalize
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Shallow MLP projector (one tower of the bi-encoder)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        # Setting up the params in the parent class
        super().__init__()
        if in_dim < 1 or out_dim < 1:
            raise ValueError(f"in_dim/out_dim must be >= 1, got {in_dim}, {out_dim}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] -> L2-normalised [..., out_dim]."""
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected last dim {self.in_dim}, got {tuple(x.shape)}"
            )
        z = self.net(x)
        # L2-normalisation
        return F.normalize(z, p=2, dim=-1)
