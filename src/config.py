"""Single source of truth for project configuration."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EmbedConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    pool_mode: str = "mean"  # "mean" or "last_token"
    dtype: torch.dtype = torch.float32
    max_length: int | None = None


@dataclass(frozen=True)
class HeadConfig:
    in_dim: int = 1024  # Qwen3-0.6B hidden size
    out_dim: int = 512
    init_scale: float = 10.0
