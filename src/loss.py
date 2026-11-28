"""Masked option cross-entropy (InfoNCE over each state's own candidate options).

Given:
  scores     [B, N] scaled cosine scores between B states and N unique options
  mask       [B, N] bool, True where option j belongs to state i
  target_idx [B]    long, column index in [0, N) of state i's correct option

Positions where mask is False are filled with -inf before cross-entropy so each
row's softmax only competes over that state's own options.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedOptionLoss(nn.Module):
    """Cross-entropy over per-state valid options using a [B, N] boolean mask."""

    def forward(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        if scores.ndim != 2:
            raise ValueError(f"expected 2D scores [B, N], got {tuple(scores.shape)}")
        # Mask shape should match the scores and also its datatype should be boolean
        if mask.shape != scores.shape or mask.dtype != torch.bool:
            raise ValueError(
                f"expected bool mask of shape {tuple(scores.shape)}, "
                f"got {tuple(mask.shape)} ({mask.dtype})"
            )
        B = scores.shape[0]
        if target_idx.shape != (B,):
            raise ValueError(f"expected target_idx of shape ({B},), got {tuple(target_idx.shape)}")
        if not mask.any(dim=1).all():
            raise ValueError("every row in mask must have at least one valid option")

        batch_idx = torch.arange(B, device=scores.device)
        if not mask[batch_idx, target_idx].all():
            raise ValueError("every target_idx must point to a True entry in mask")

        # Filling all the false mask values with -inf
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        return F.cross_entropy(masked_scores, target_idx)
