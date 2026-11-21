"""Day 4: compare mean vs last_token pooling on hand-labeled pairs.

Expectation:
  match pairs     → high cosine
  non-match pairs → low cosine
  gap = avg(match) - avg(non_match)  → bigger is better
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embed import embed_texts  # noqa: E402

# 10 should match, 10 should not. Hand-written on purpose — not from bev.
MATCH_PAIRS: list[tuple[str, str]] = [
    (
        "The kitchen timer beeped twice.",
        "A timer in the kitchen rang two times.",
    ),
    (
        "She booked a train for Tuesday morning.",
        "She reserved a Tuesday-morning train ticket.",
    ),
    (
        "Rain tapped softly on the window.",
        "Soft rain hit the windowpane.",
    ),
    (
        "He chose the quieter route home.",
        "He took the less noisy way back home.",
    ),
    (
        "The bakery sold out of sourdough early.",
        "Sourdough was gone from the bakery by morning.",
    ),
    (
        "They postponed the meeting until Friday.",
        "The meeting was delayed to Friday.",
    ),
    (
        "Fog blurred the harbor lights at dawn.",
        "At sunrise, fog made the harbor lights hard to see.",
    ),
    (
        "Someone left a map on the cafe table.",
        "A map was left behind on the coffee-shop table.",
    ),
    (
        "The bike chain needed oil again.",
        "His bicycle chain required more lubricant.",
    ),
    (
        "Two sparrows argued over a crumb.",
        "A pair of sparrows fought about a scrap of food.",
    ),
]

NON_MATCH_PAIRS: list[tuple[str, str]] = [
    (
        "The kitchen timer beeped twice.",
        "Thunder rolled past the distant hills.",
    ),
    (
        "She booked a train for Tuesday morning.",
        "A stray cat watched from the fence.",
    ),
    (
        "Rain tapped softly on the window.",
        "The API returned a calibrated confidence score.",
    ),
    (
        "He chose the quieter route home.",
        "Batch size sixteen fit in memory.",
    ),
    (
        "The bakery sold out of sourdough early.",
        "Hidden states were cached to disk once.",
    ),
    (
        "They postponed the meeting until Friday.",
        "Cosine similarity picked the wrong option.",
    ),
    (
        "Fog blurred the harbor lights at dawn.",
        "She labeled the eval split with seed forty-two.",
    ),
    (
        "Someone left a map on the cafe table.",
        "The model scored every option in one pass.",
    ),
    (
        "The bike chain needed oil again.",
        "Leaves spun across the empty courtyard.",
    ),
    (
        "Two sparrows argued over a crumb.",
        "Coffee cooled beside an open notebook.",
    ),
]

MODES = ("mean", "last_token")


def cosine_pairs(
    left_vecs: torch.Tensor,
    right_vecs: torch.Tensor,
) -> torch.Tensor:
    """Pairwise cosine for aligned rows: [N] from [N, H] x [N, H]."""
    return F.cosine_similarity(left_vecs.float(), right_vecs.float(), dim=-1)


def score_mode(mode: str) -> dict[str, float | list[float]]:
    match_left, match_right = zip(*MATCH_PAIRS)
    non_left, non_right = zip(*NON_MATCH_PAIRS)

    # One forward batch per side keeps padding consistent within the side
    m_l = embed_texts(list(match_left), mode=mode)
    m_r = embed_texts(list(match_right), mode=mode)
    n_l = embed_texts(list(non_left), mode=mode)
    n_r = embed_texts(list(non_right), mode=mode)

    match_cos = cosine_pairs(m_l, m_r)
    non_cos = cosine_pairs(n_l, n_r)

    avg_match = match_cos.mean().item()
    avg_non = non_cos.mean().item()
    return {
        "match_cosines": match_cos.tolist(),
        "non_match_cosines": non_cos.tolist(),
        "avg_match": avg_match,
        "avg_non_match": avg_non,
        "gap": avg_match - avg_non,
    }


def main() -> None:
    print(f"pairs: {len(MATCH_PAIRS)} match + {len(NON_MATCH_PAIRS)} non-match")
    print(f"{'mode':<12}  {'avg_match':>10}  {'avg_non':>10}  {'gap':>10}")
    print("-" * 48)

    results: dict[str, dict] = {}
    for mode in MODES:
        results[mode] = score_mode(mode)
        r = results[mode]
        print(
            f"{mode:<12}  {r['avg_match']:10.4f}  {r['avg_non_match']:10.4f}  "
            f"{r['gap']:10.4f}"
        )

    print("\n--- per-pair cosines ---")
    for mode in MODES:
        r = results[mode]
        print(f"\n[{mode}] MATCH")
        for (a, b), c in zip(MATCH_PAIRS, r["match_cosines"]):
            print(f"  {c:7.4f}  {a[:42]!r}  ↔  {b[:42]!r}")
        print(f"[{mode}] NON-MATCH")
        for (a, b), c in zip(NON_MATCH_PAIRS, r["non_match_cosines"]):
            print(f"  {c:7.4f}  {a[:42]!r}  ↔  {b[:42]!r}")

    winner = max(MODES, key=lambda m: results[m]["gap"])
    print(
        f"\nWinner (bigger gap): {winner}  "
        f"(gap={results[winner]['gap']:.4f})"
    )
    print("Use this pool mode for Day 5 closest-choice eval.")


if __name__ == "__main__":
    main()
