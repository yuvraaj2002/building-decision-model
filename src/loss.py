"""Day 9: InfoNCE loss over a [B, B] scaled similarity matrix.

For a batch of B (state, positive_action) pairs:
  scores[i, j] = cosine(state_i, action_j) * scale
  diagonal (i == j) = positive pairs
  off-diagonal (i != j) = in-batch negatives

InfoNCE is cross-entropy across each row with target label i.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from heads import BiEncoderHeads, DEFAULT_IN_DIM, DEFAULT_OUT_DIM


def infonce_loss(scores: torch.Tensor, symmetric: bool = False) -> torch.Tensor:
    """InfoNCE loss on a [B, B] scaled similarity matrix (diagonal = positives).

    Args:
        scores: [B, B] matrix where row i is state i scored against all B actions.
        symmetric: if True, average state->action and action->state (CLIP-style).
            False matches Sentence-Transformers MultipleNegativesRankingLoss.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"expected square [B, B] matrix, got {tuple(scores.shape)}")

    labels = torch.arange(scores.shape[0], device=scores.device)
    loss_s2a = F.cross_entropy(scores, labels)
    if not symmetric:
        return loss_s2a
    loss_a2s = F.cross_entropy(scores.T, labels)
    return 0.5 * (loss_s2a + loss_a2s)


def _run_unit_checks() -> None:
    torch.manual_seed(0)
    B, H, D = 16, DEFAULT_IN_DIM, DEFAULT_OUT_DIM
    model = BiEncoderHeads(in_dim=H, out_dim=D, init_scale=10.0)
    states = torch.randn(B, H)
    actions = torch.randn(B, H)

    s = model.encode_states(states)
    a = model.encode_actions(actions)
    scores = model.score_matrix(states, actions)

    loss_val = infonce_loss(scores, symmetric=False)
    loss_sym = infonce_loss(scores, symmetric=True)
    expected_random = math.log(B)

    # Perfect-match sanity check: diagonal cosine = 1.0, off-diagonal ~ 0.
    perfect_scores = model.scale * (s @ s.T)
    perfect_loss = infonce_loss(perfect_scores, symmetric=False)
    assert perfect_loss.item() < 0.01, f"expected near-zero loss, got {perfect_loss.item()}"

    print("loss unit checks OK")
    print(f"  batch size B:         {B}")
    print(f"  random InfoNCE:       {loss_val.item():.6f}  (log(B) = {expected_random:.6f})")
    print(f"  symmetric InfoNCE:    {loss_sym.item():.6f}")
    print(f"  perfect-match loss:   {perfect_loss.item():.6f}")

    # Optional reference check against Sentence-Transformers MultipleNegativesRankingLoss.
    try:
        from sentence_transformers.losses import MultipleNegativesRankingLoss
    except ImportError:
        print("  MNRL reference check: skipped (run with `uv run --with sentence-transformers python src/loss.py`)")
        return

    class _DummyModel(torch.nn.Module):
        def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return features

    mnrl = MultipleNegativesRankingLoss(model=_DummyModel(), scale=model.scale.item())
    ref_loss = mnrl([{"sentence_embedding": s}, {"sentence_embedding": a}], labels=None)
    diff = (loss_val - ref_loss).abs().item()
    assert torch.allclose(loss_val, ref_loss, atol=1e-6), (loss_val.item(), ref_loss.item(), diff)
    print(f"  MNRL reference loss:  {ref_loss.item():.6f}  (abs diff = {diff:.2e})")


if __name__ == "__main__":
    _run_unit_checks()
