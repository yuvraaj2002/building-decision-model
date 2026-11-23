"""Day 8: bi-encoder projection heads on frozen Qwen vectors.

Each head remaps ℝ^{in_dim} → ℝ^{out_dim}:
  Linear → LayerNorm → GELU → Linear → L2-normalize

BiEncoderHeads keeps two towers (state / action) plus a learned scale:
  score_matrix[i, j] = cosine(state_i, action_j) × scale
which is [B, B] for a batch of B pairs (diagonal = positives for InfoNCE).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_IN_DIM = 1024  # Qwen3-0.6B hidden size
DEFAULT_OUT_DIM = 512


class ProjectionHead(nn.Module):
    """Shallow MLP projector (one tower of the bi-encoder)."""

    def __init__(self, in_dim: int = DEFAULT_IN_DIM, out_dim: int = DEFAULT_OUT_DIM):
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
        """x: [..., in_dim] → L2-normalised [..., out_dim]."""
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected last dim {self.in_dim}, got {tuple(x.shape)}"
            )
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class BiEncoderHeads(nn.Module):
    """State head + action head + learned cosine scale."""

    def __init__(
        self,
        in_dim: int = DEFAULT_IN_DIM,
        out_dim: int = DEFAULT_OUT_DIM,
        init_scale: float = 10.0,
    ):
        super().__init__()
        self.state_head = ProjectionHead(in_dim=in_dim, out_dim=out_dim)
        self.action_head = ProjectionHead(in_dim=in_dim, out_dim=out_dim)
        # Store log-scale so scale = exp(logit_scale) stays positive (CLIP-style).
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    @property
    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp()

    def encode_states(self, states: torch.Tensor) -> torch.Tensor:
        return self.state_head(states)

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.action_head(actions)

    def score_matrix(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B, B] scores: rows = states, cols = actions.

        states/actions: [B, in_dim] frozen Qwen vectors (or already any batch).
        After L2-norm, cosine = dot product.
        """
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                f"expected 2D batches, got states {tuple(states.shape)} "
                f"actions {tuple(actions.shape)}"
            )
        if states.shape[0] != actions.shape[0]:
            raise ValueError(
                f"batch sizes must match for [B,B] matrix, got "
                f"{states.shape[0]} vs {actions.shape[0]}"
            )
        s = self.encode_states(states)
        a = self.encode_actions(actions)
        return self.scale * (s @ a.T)

    def score_pairs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Per-row cosine × scale for aligned pairs → [B]."""
        s = self.encode_states(states)
        a = self.encode_actions(actions)
        cos = (s * a).sum(dim=-1)
        return self.scale * cos


def _run_unit_checks() -> None:
    torch.manual_seed(0)
    B, H, D = 4, DEFAULT_IN_DIM, DEFAULT_OUT_DIM
    model = BiEncoderHeads(in_dim=H, out_dim=D)
    states = torch.randn(B, H)
    actions = torch.randn(B, H)

    s = model.encode_states(states)
    a = model.encode_actions(actions)
    assert s.shape == (B, D), s.shape
    assert a.shape == (B, D), a.shape

    s_norms = s.norm(dim=-1)
    a_norms = a.norm(dim=-1)
    assert torch.allclose(s_norms, torch.ones(B), atol=1e-5), s_norms
    assert torch.allclose(a_norms, torch.ones(B), atol=1e-5), a_norms

    scores = model.score_matrix(states, actions)
    assert scores.shape == (B, B), scores.shape

    paired = model.score_pairs(states, actions)
    assert paired.shape == (B,), paired.shape
    diag = scores.diag()
    assert torch.allclose(paired, diag, atol=1e-5), (paired, diag)

    n_params = sum(p.numel() for p in model.parameters())
    print("heads unit checks OK")
    print(f"  state/action out: {tuple(s.shape)}")
    print(f"  score_matrix:     {tuple(scores.shape)}")
    print(f"  L2 norms ~ 1:     state {s_norms.mean().item():.4f}  action {a_norms.mean().item():.4f}")
    print(f"  scale:            {model.scale.item():.4f}")
    print(f"  params:           {n_params:,}")


if __name__ == "__main__":
    _run_unit_checks()
