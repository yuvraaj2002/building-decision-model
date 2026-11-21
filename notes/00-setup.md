# What running a 0.6B model on a MacBook Air actually costs, in milliseconds

**Date:** 2026-11-21  
**Machine:** MacBook Air, Apple Silicon, 24 GB unified memory  
**Model:** `Qwen/Qwen3-0.6B` on MPS (`hidden_size=1024`)

## Day 1 — is the GPU even usable?

`scripts/check_mps.py` (4096×4096 matmul):

| Device | Time |
|---|---|
| CPU | ~169 ms |
| MPS | ~123 ms |

MPS is available and faster than CPU for dense matmul. Good enough to proceed.

## Day 2 — first real embedding forward

`src/embed.py` loads Qwen3-0.6B, embeds 20 short sentences, and times a forward with `output_hidden_states=True` after a warmup + `torch.mps.synchronize()`.

Hidden-state shape is `[batch, seq_len, 1024]` (seq_len is the padded max length in that batch).

| batch | fp32 ms/sent | fp16 ms/sent |
|---|---|---|
| 1 | 48.0 | 74.4 |
| 8 | 19.0 | 14.5 |
| 16 | 16.1 | 18.4 |

Takeaways:

- Larger batches amortize launch overhead in fp32 (48 → 16 ms/sent from batch 1 → 16).
- fp16 helps at batch 8, but batch 1 / 16 were not consistently faster than fp32 in this run — treat fp16 on MPS as “try it, measure it,” not automatic win.
- Model weights are ~1.5 GB on disk; first download dominated the session, not inference.
- Tokenizer alone gives `[B, L]` IDs; the `1024` dim only appears after the model (or embedding lookup).

## Learnings along the way

- `output_hidden_states` belongs on the **forward** call, not `from_pretrained`.
- Tokenizer returns a `BatchEncoding` (dict of tensors), not a tensor — use `batch["input_ids"].shape`.
- With `padding=True`, shape is `[B, max_len_in_batch]`.
- Move every tensor in the batch to MPS before `model(**batch)`.
- Sync MPS around timers or the number lies.

Milestone 0 done. Next: zero-shot baseline (Day 3 — load decision datasets).
