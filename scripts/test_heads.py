"""Test script: verify ProjectionHead and BiEncoderHeads shapes, norms, and scoring.

Run:
    uv run python scripts/test_heads.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bi_encoder import BiEncoderHeads  # noqa: E402
from config import HeadConfig  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    cfg = HeadConfig()
    B = 4

    model = BiEncoderHeads(cfg)
    states = torch.randn(B, cfg.in_dim)
    actions = torch.randn(B, cfg.in_dim)

    s = model.encode_states(states)
    a = model.encode_actions(actions)
    assert s.shape == (B, cfg.out_dim), s.shape
    assert a.shape == (B, cfg.out_dim), a.shape

    s_norms = s.norm(dim=-1)
    a_norms = a.norm(dim=-1)
    assert torch.allclose(s_norms, torch.ones(B), atol=1e-5), s_norms
    assert torch.allclose(a_norms, torch.ones(B), atol=1e-5), a_norms

    scores = model.score_matrix(states, actions)
    assert scores.shape == (B, B), scores.shape

    paired = model.score_pairs(states, actions)
    assert paired.shape == (B,), paired.shape
    assert torch.allclose(paired, scores.diag(), atol=1e-5), (paired, scores.diag())

    n_params = sum(p.numel() for p in model.parameters())
    print("PASS: heads unit checks")
    print(f"  state/action out: {tuple(s.shape)}")
    print(f"  score_matrix:     {tuple(scores.shape)}")
    print(
        f"  L2 norms ~ 1:     state {s_norms.mean().item():.4f}  "
        f"action {a_norms.mean().item():.4f}"
    )
    print(f"  scale:            {model.scale.item():.4f}")
    print(f"  params:           {n_params:,}")


if __name__ == "__main__":
    main()
