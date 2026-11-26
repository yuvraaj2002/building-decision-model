"""Test script: embed a batch of 32 training examples and inspect the [B, N] raw cosine matrix + mask.

Run:
    uv run python scripts/test_option_scores.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bi_encoder import BiEncoderHeads  # noqa: E402
from config import EmbedConfig, HeadConfig  # noqa: E402
from data import load_bev_examples  # noqa: E402
from embed import TextEmbedder  # noqa: E402
from option_batch import OptionBatch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect [B, N] raw cosine scores and mask")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(0)

    print(f"Loading {args.batch_size} training examples from avbiswas/bev-decision...")
    examples = load_bev_examples(split="train", limit=args.batch_size)
    batch = OptionBatch.from_examples(examples)

    B = len(batch.state_texts)
    N = len(batch.unique_options)

    embedder = TextEmbedder(EmbedConfig(max_length=args.max_length))
    state_vecs = embedder.embed(batch.state_texts)
    option_vecs = embedder.embed(batch.unique_options)

    model = BiEncoderHeads(HeadConfig()).to(device=embedder.device, dtype=embedder.dtype)
    model.eval()
    with torch.no_grad():
        raw_scores = model.score_options(state_vecs, option_vecs, scaled=False)

    assert raw_scores.shape == (B, N), raw_scores.shape
    assert batch.mask.shape == (B, N), batch.mask.shape
    assert batch.target_idx.shape == (B,), batch.target_idx.shape
    assert raw_scores.min().item() >= -1.0001 and raw_scores.max().item() <= 1.0001

    print("\nPASS: [B, N] raw cosine scores + mask")
    print(f"  B (states):           {B}")
    print(f"  total options:        {batch.total_options} -> deduplicated N = {N}")
    print(f"  state_vecs shape:     {tuple(state_vecs.shape)}")
    print(f"  option_vecs shape:    {tuple(option_vecs.shape)}")
    print(f"  raw_scores shape:     {tuple(raw_scores.shape)}  (min={raw_scores.min().item():.4f}, max={raw_scores.max().item():.4f})")
    print(f"  mask shape:           {tuple(batch.mask.shape)}  (True count={int(batch.mask.sum().item())})")
    print(f"  target_idx shape:     {tuple(batch.target_idx.shape)}")

    print("\n--- First 3 states ---")
    for i in range(min(3, B)):
        valid_cols = batch.mask[i].nonzero(as_tuple=True)[0].tolist()
        valid_scores = [round(raw_scores[i, j].item(), 4) for j in valid_cols]
        gold_col = int(batch.target_idx[i].item())
        print(f"  state[{i}]: {len(valid_cols)} options | cols={valid_cols} | raw_cos={valid_scores} | target_col={gold_col}")


if __name__ == "__main__":
    main()
