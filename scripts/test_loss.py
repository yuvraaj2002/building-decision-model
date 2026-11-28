"""Test script: verify MaskedOptionLoss on variable-option batches.

Run:
    uv run python scripts/test_loss.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bi_encoder import BiEncoderHeads  # noqa: E402
from config import HeadConfig  # noqa: E402
from loss import MaskedOptionLoss  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    criterion = MaskedOptionLoss()

    # 3 states with 2, 3, and 4 options (N = 9 total columns)
    mask = torch.tensor(
        [
            [True, True, False, False, False, False, False, False, False],
            [False, False, True, True, True, False, False, False, False],
            [False, False, False, False, False, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    target_idx = torch.tensor([0, 4, 6], dtype=torch.long)

    # 1. Uniform scores -> loss must equal average of log(k_i) = mean(log(2), log(3), log(4))
    uniform_scores = torch.zeros((3, 9))
    uniform_loss = criterion(uniform_scores, mask, target_idx)
    expected_uniform = (math.log(2) + math.log(3) + math.log(4)) / 3.0
    assert torch.isclose(uniform_loss, torch.tensor(expected_uniform), atol=1e-6), (
        uniform_loss.item(),
        expected_uniform,
    )

    # 2. Masked-out columns must have zero effect on loss and zero gradient
    scores = torch.randn((3, 9), requires_grad=True)
    loss_a = criterion(scores, mask, target_idx)
    loss_a.backward()
    assert scores.grad is not None
    assert torch.all(scores.grad[~mask] == 0.0), "masked-out columns received non-zero grad"

    perturbed = scores.detach().clone()
    perturbed[~mask] = 999.0
    loss_b = criterion(perturbed, mask, target_idx)
    assert torch.isclose(loss_a.detach(), loss_b, atol=1e-6), (loss_a.item(), loss_b.item())

    # 3. Perfect scores on target columns -> near-zero loss
    perfect_scores = torch.full((3, 9), -10.0)
    perfect_scores[torch.arange(3), target_idx] = 10.0
    perfect_loss = criterion(perfect_scores, mask, target_idx)
    assert perfect_loss.item() < 1e-6, perfect_loss.item()

    # 4. End-to-end check with BiEncoderHeads.score_options(scaled=True)
    cfg = HeadConfig()
    model = BiEncoderHeads(cfg)
    state_vecs = torch.randn(3, cfg.in_dim)
    option_vecs = torch.randn(9, cfg.in_dim)
    scaled_scores = model.score_options(state_vecs, option_vecs, scaled=True)
    model_loss = criterion(scaled_scores, mask, target_idx)
    model_loss.backward()
    assert model.logit_scale.grad is not None

    print("PASS: MaskedOptionLoss unit checks")
    print(f"  uniform loss:      {uniform_loss.item():.6f}  (expected {expected_uniform:.6f})")
    print(f"  masked-col grad=0: True")
    print(f"  perfect loss:      {perfect_loss.item():.6f}")
    print(f"  bi-encoder loss:   {model_loss.item():.6f}")


if __name__ == "__main__":
    main()
