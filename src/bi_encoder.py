"""Bi-encoder model combining state/action projection heads and a learned scale.

  score_matrix[i, j] = cosine(state_i, action_j) * scale
which is [B, B] for a batch of B pairs (diagonal = positives for InfoNCE).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from config import HeadConfig
from heads import ProjectionHead


class BiEncoderHeads(nn.Module):
    """State head + action head + learned cosine scale."""

    def __init__(self, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadConfig()
        self.state_head = ProjectionHead(
            in_dim=self.config.in_dim, out_dim=self.config.out_dim
        )
        self.action_head = ProjectionHead(
            in_dim=self.config.in_dim, out_dim=self.config.out_dim
        )
        # Store log-scale so scale = exp(logit_scale) stays positive (CLIP-style).
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(self.config.init_scale))
        )

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

        After L2-norm, cosine = dot product.
        """
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                f"expected 2D batches, got states {tuple(states.shape)} "
                f"actions {tuple(actions.shape)}"
            )
        if states.shape[0] != actions.shape[0]:
            raise ValueError(
                f"batch sizes must match for [B, B] matrix, got "
                f"{states.shape[0]} vs {actions.shape[0]}"
            )
        s = self.encode_states(states)
        a = self.encode_actions(actions)
        return self.scale * (s @ a.T)

    def score_options(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        *,
        scaled: bool = False,
    ) -> torch.Tensor:
        """Return [B, N] cosine scores between B states and N unique options."""
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                f"expected 2D batches, got states {tuple(states.shape)} "
                f"actions {tuple(actions.shape)}"
            )
        s = self.encode_states(states)
        a = self.encode_actions(actions)
        cos = s @ a.T
        return self.scale * cos if scaled else cos

    def score_pairs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Per-row cosine * scale for aligned pairs -> [B]."""
        s = self.encode_states(states)
        a = self.encode_actions(actions)
        cos = (s * a).sum(dim=-1)
        return self.scale * cos
