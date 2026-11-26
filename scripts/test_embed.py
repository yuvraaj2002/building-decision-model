"""Test script: generate embeddings for a batch of 10 strings using TextEmbedder.

Run:
    uv run python scripts/test_embed.py
    uv run python scripts/test_embed.py --model Qwen/Qwen3-Embedding-0.6B --pool last_token
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import EmbedConfig  # noqa: E402
from embed import TextEmbedder  # noqa: E402

SAMPLE_STRINGS = [
    "The kitchen timer beeped twice.",
    "Rain tapped softly on the window.",
    "She booked a train for Tuesday morning.",
    "The model scored every option in one pass.",
    "Coffee cooled beside an open notebook.",
    "He chose the quieter route home.",
    "A stray cat watched from the fence.",
    "The API returned a calibrated confidence score.",
    "Leaves spun across the empty courtyard.",
    "They postponed the meeting until Friday.",
]


def main() -> None:
    defaults = EmbedConfig()
    parser = argparse.ArgumentParser(description="Embed a batch of 10 strings.")
    parser.add_argument("--model", default=defaults.model_name, help="HF model name")
    parser.add_argument("--pool", default=defaults.pool_mode, choices=["mean", "last_token"])
    parser.add_argument("--fp16", action="store_true", help="Run in float16")
    parser.add_argument("--max-length", type=int, default=defaults.max_length)
    args = parser.parse_args()

    config = EmbedConfig(
        model_name=args.model,
        pool_mode=args.pool,
        dtype=torch.float16 if args.fp16 else defaults.dtype,
        max_length=args.max_length,
    )
    embedder = TextEmbedder(config)

    embedder.sync()
    start = time.monotonic()
    vectors = embedder.embed(SAMPLE_STRINGS)
    embedder.sync()
    elapsed_ms = (time.monotonic() - start) * 1000

    assert vectors.shape == (len(SAMPLE_STRINGS), embedder.hidden_size), (
        f"Expected {(len(SAMPLE_STRINGS), embedder.hidden_size)}, got {tuple(vectors.shape)}"
    )
    assert not torch.isnan(vectors).any(), "Embeddings contain NaNs!"

    print(
        f"model={embedder.model_name}  device={embedder.device}  "
        f"dtype={embedder.dtype}  pool={embedder.pool_mode}"
    )
    print(
        f"batch={len(SAMPLE_STRINGS)}  shape={tuple(vectors.shape)}  "
        f"total={elapsed_ms:.2f} ms  ({elapsed_ms / len(SAMPLE_STRINGS):.2f} ms/str)\n"
    )
    for i, (text, vec) in enumerate(zip(SAMPLE_STRINGS, vectors), start=1):
        norm = vec.float().norm().item()
        print(f"  {i:>2}. norm={norm:6.2f}  {text}")


if __name__ == "__main__":
    main()
