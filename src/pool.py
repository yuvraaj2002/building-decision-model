"""Collapse token hidden states into one vector per text."""

from __future__ import annotations

import torch


def pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """Pool last-layer hidden states to one vector per sequence.

    Args:
        hidden: [B, L, H] token representations
        attention_mask: [B, L] 1 = real token, 0 = padding
        mode: "mean" | "last_token"

    Returns:
        [B, H] one representation per text
    """
    if mode == "mean":
        # Sum only real tokens, divide by their count (avoid pad pollution)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # [B, L, 1]
        summed = (hidden * mask).sum(dim=1)  # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        return summed / counts

    if mode == "last_token":
        # Index of last non-pad token per row (left-padded or right-padded safe)
        lengths = attention_mask.sum(dim=1) - 1  # [B]
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]  # [B, H]

    raise ValueError(f"unknown pool mode: {mode!r} (use 'mean' or 'last_token')")
