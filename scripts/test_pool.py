"""Test script: verify Pooler (mean & last_token) on right-padded and left-padded batches.

Run:
    uv run python scripts/test_pool.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pool import Pooler  # noqa: E402


def main() -> None:
    # Batch of 2 sequences, length 4, hidden_size 3
    # Row 0: 2 real tokens + 2 pad tokens (right-padded)
    # Row 1: 4 real tokens
    hidden_right = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [99.0, 99.0, 99.0], [99.0, 99.0, 99.0]],
            [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        ]
    )
    mask_right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])

    # Same real tokens, but left-padded for row 0
    hidden_left = torch.tensor(
        [
            [[99.0, 99.0, 99.0], [99.0, 99.0, 99.0], [1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        ]
    )
    mask_left = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])

    expected_mean = torch.tensor([[2.0, 3.0, 4.0], [5.0, 0.0, 0.0]])
    expected_last = torch.tensor([[3.0, 4.0, 5.0], [8.0, 0.0, 0.0]])

    mean_pooler = Pooler("mean")
    assert torch.allclose(mean_pooler.pool(hidden_right, mask_right), expected_mean)
    assert torch.allclose(mean_pooler.pool(hidden_left, mask_left), expected_mean)
    print("PASS: mean pooling (right-padded & left-padded)")

    last_pooler = Pooler("last_token")
    assert torch.allclose(last_pooler.pool(hidden_right, mask_right), expected_last)
    assert torch.allclose(last_pooler.pool(hidden_left, mask_left), expected_last)
    print("PASS: last_token pooling (right-padded & left-padded)")

    try:
        Pooler("invalid")
        raise AssertionError("Expected ValueError for invalid mode")
    except ValueError:
        print("PASS: invalid mode raises ValueError")


if __name__ == "__main__":
    main()
