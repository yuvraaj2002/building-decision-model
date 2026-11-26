"""Collapse token hidden states into one vector per text."""

from __future__ import annotations

import torch


class Pooler:
    """Pool last-layer token hidden states [B, L, H] into one vector per text [B, H]."""

    VALID_MODES = ("mean", "last_token")

    def __init__(self, mode: str = "mean") -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"unknown pool mode: {mode!r} (use one of {self.VALID_MODES})"
            )
        self.mode = mode

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool [B, L, H] hidden states using [B, L] attention_mask (1 = real, 0 = pad)."""
        if self.mode == "mean":
            return self._mean(hidden, attention_mask)
        return self._last_token(hidden, attention_mask)

    @staticmethod
    def _mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, L, 1]
        summed = (hidden * mask).sum(dim=1)  # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        return summed / counts

    @staticmethod
    def _last_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Works for both right-padded ([1, 1, 0, 0]) and left-padded ([0, 0, 1, 1]) batches
        seq_len = attention_mask.size(1)
        positions = torch.arange(seq_len, device=hidden.device).unsqueeze(0)  # [1, L]
        last_idx = (positions * (attention_mask > 0).long()).max(dim=1).values  # [B]
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, last_idx]  # [B, H]
