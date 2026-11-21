"""Embed texts with Qwen3-0.6B → one vector per text (pooled hidden states)."""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pool import pool

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to(device)
model.eval()

sample_sentences = [
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
    "Hidden states were cached to disk once.",
    "The bakery sold out of sourdough early.",
    "Fog blurred the harbor lights at dawn.",
    "Batch size sixteen fit in memory.",
    "Someone left a map on the cafe table.",
    "The bike chain needed oil again.",
    "Cosine similarity picked the wrong option.",
    "Thunder rolled past the distant hills.",
    "She labeled the eval split with seed forty-two.",
    "Two sparrows argued over a crumb.",
]

batch_sizes = [1, 8, 16]
precisions = ["fp32", "fp16"]
POOL_MODE = "mean"  # or "last_token"


def sync():
    if device == "mps":
        torch.mps.synchronize()


def embed_texts(
    texts: list[str],
    *,
    mode: str = POOL_MODE,
    dtype: torch.dtype = torch.float32,
    max_length: int | None = None,
) -> torch.Tensor:
    """Return one vector per text: [B, hidden_size].

    If max_length is set, truncate (and pad) to that many tokens — critical
    for long BEV states on Mac so batches stay in memory and finish fast.
    """
    model.to(dtype=dtype)
    tok_kwargs: dict = {"return_tensors": "pt", "padding": True}
    if max_length is not None:
        tok_kwargs["truncation"] = True
        tok_kwargs["max_length"] = max_length
    batch = tokenizer(texts, **tok_kwargs)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
        hidden = out.hidden_states[-1]  # [B, L, H]
        return pool(hidden, batch["attention_mask"], mode=mode)


def run_inference(batch_size: int, precision: str, mode: str = POOL_MODE):
    """Tokenize, forward, pool → one vec/text; return shapes + timings."""
    dtype = torch.float16 if precision == "fp16" else torch.float32
    texts = sample_sentences[:batch_size]

    # Warmup
    _ = embed_texts(texts, mode=mode, dtype=dtype)
    sync()

    sync()
    start = time.monotonic()
    vectors = embed_texts(texts, mode=mode, dtype=dtype)
    sync()
    elapsed_ms = (time.monotonic() - start) * 1000

    # Also report raw hidden shape for comparison
    batch = tokenizer(texts, return_tensors="pt", padding=True)
    seq_len = batch["input_ids"].shape[1]
    hidden_shape = (batch_size, seq_len, model.config.hidden_size)
    return hidden_shape, tuple(vectors.shape), elapsed_ms, elapsed_ms / batch_size


if __name__ == "__main__":
    print(f"device={device}  hidden_size={model.config.hidden_size}  pool={POOL_MODE}")
    print(
        f"{'batch':>6}  {'prec':>5}  {'token_hidden':>18}  {'text_vec':>14}  "
        f"{'total_ms':>10}  {'ms/sent':>10}"
    )

    for precision in precisions:
        for batch_size in batch_sizes:
            hidden_shape, vec_shape, total_ms, ms_per_sent = run_inference(
                batch_size, precision
            )
            print(
                f"{batch_size:>6}  {precision:>5}  {str(hidden_shape):>18}  "
                f"{str(vec_shape):>14}  {total_ms:>10.2f}  {ms_per_sent:>10.2f}"
            )

    # Demo: 3 texts → 3 vectors
    demo = sample_sentences[:3]
    vecs = embed_texts(demo, mode=POOL_MODE, dtype=torch.float32)
    print(f"\nDemo: {len(demo)} texts → vectors shape {tuple(vecs.shape)}")
    for text, v in zip(demo, vecs):
        print(f"  {tuple(v.shape)}  {text[:48]!r}")

    model.to(dtype=torch.float32)
