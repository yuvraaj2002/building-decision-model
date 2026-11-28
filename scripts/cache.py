"""Day 7: cache frozen Qwen embeddings for bi-encoder head training.

Encodes train rows once (states and options separately) with the M1-winning
setup: labeled prompts, mean pooling, fp16, max_length 256.

Writes under cache/:
  states.pt       — [N, H] query vectors (Situation + Question)
  options.pt      — [U, H] unique option vectors (Option: …)
  positive_idx.pt — [N] long; options[positive_idx[i]] is the correct choice
  rows.jsonl      — row metadata (qtype, choices, correct_idx, option ids)
  meta.json       — config, timing, disk size, shapes

Eval rows in data/eval_bev.jsonl are never included.

Usage:
  uv run python scripts/cache.py --limit 20          # smoke
  uv run python scripts/cache.py --n 5000            # Day 7 target
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import EmbedConfig  # noqa: E402
from data import EVAL_PATH, iter_bev_examples, load_jsonl, save_jsonl  # noqa: E402
from embed import TextEmbedder  # noqa: E402
from option_batch import OptionBatch  # noqa: E402

CACHE_DIR = ROOT / "cache"
DEFAULT_N = 5000
DEFAULT_SEED = 42
DEFAULT_EMBED_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 256
PROMPT_FORMAT = "labeled"


def example_key(ex: dict) -> tuple:
    return (ex["state"], ex["question"], tuple(ex["choices"]), ex["correct_idx"])


def collect_train_examples(
    n: int,
    *,
    seed: int,
    buffer_mult: int = 3,
) -> list[dict]:
    """Sample n train examples, excluding the seeded eval split."""
    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"missing {EVAL_PATH}; run `uv run python src/data.py` first"
        )
    eval_keys = {example_key(e) for e in load_jsonl(EVAL_PATH)}
    want = max(n * buffer_mult, n)
    buf: list[dict] = []
    for ex in iter_bev_examples(split="train"):
        if example_key(ex) in eval_keys:
            continue
        buf.append(ex)
        if len(buf) >= want:
            break
    if len(buf) < n:
        raise RuntimeError(f"only collected {len(buf)} train examples, need {n}")
    rng = random.Random(seed)
    return rng.sample(buf, n)


def embed_unique(
    embedder: TextEmbedder,
    texts: list[str],
    *,
    embed_batch_size: int,
    label: str,
) -> torch.Tensor:
    """Embed a list of unique strings in batches → [len(texts), H] on CPU fp16."""
    if not texts:
        raise ValueError(f"{label}: empty text list")
    parts: list[torch.Tensor] = []
    total = len(texts)
    for i in range(0, total, embed_batch_size):
        chunk = texts[i : i + embed_batch_size]
        vecs = embedder.embed(chunk)
        parts.append(vecs.detach().to(device="cpu", dtype=torch.float16))
        done = min(i + embed_batch_size, total)
        if done == total or done % (embed_batch_size * 10) < embed_batch_size:
            print(f"  {label}: {done}/{total}", flush=True)
    embedder.sync()
    return torch.cat(parts, dim=0)


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def format_bytes(n: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n} B"


def build_cache(
    examples: list[dict],
    *,
    out_dir: Path,
    embed_batch_size: int,
    max_length: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    query_texts = [OptionBatch._format_state(ex) for ex in examples]

    # Unique option strings (labeled), shared across rows.
    option_to_id: dict[str, int] = {}
    unique_options: list[str] = []
    positive_idx: list[int] = []
    rows_out: list[dict] = []

    for i, ex in enumerate(examples):
        choice_ids: list[int] = []
        for c in ex["choices"]:
            labeled = OptionBatch._format_option(c)
            if labeled not in option_to_id:
                option_to_id[labeled] = len(unique_options)
                unique_options.append(labeled)
            choice_ids.append(option_to_id[labeled])
        pos = choice_ids[ex["correct_idx"]]
        positive_idx.append(pos)
        rows_out.append(
            {
                "row_id": i,
                "qtype": ex["qtype"],
                "domain": ex.get("domain", ""),
                "correct_idx": ex["correct_idx"],
                "choice_option_ids": choice_ids,
                "positive_option_id": pos,
                "n_choices": len(ex["choices"]),
            }
        )

    embed_cfg = EmbedConfig(dtype=torch.float16, max_length=max_length)
    embedder = TextEmbedder(embed_cfg)

    print(
        f"Encoding {len(query_texts)} queries + {len(unique_options)} unique options "
        f"(model={embed_cfg.model_name}, pool={embed_cfg.pool_mode}, format={PROMPT_FORMAT}, "
        f"fp16, max_length={max_length})"
    )
    embedder.sync()
    t0 = time.monotonic()

    states = embed_unique(
        embedder,
        query_texts,
        embed_batch_size=embed_batch_size,
        label="states",
    )
    options = embed_unique(
        embedder,
        unique_options,
        embed_batch_size=embed_batch_size,
        label="options",
    )
    embedder.sync()
    encode_s = time.monotonic() - t0

    pos_tensor = torch.tensor(positive_idx, dtype=torch.long)

    states_path = out_dir / "states.pt"
    options_path = out_dir / "options.pt"
    pos_path = out_dir / "positive_idx.pt"
    rows_path = out_dir / "rows.jsonl"
    meta_path = out_dir / "meta.json"
    options_txt_path = out_dir / "option_texts.json"

    torch.save(states, states_path)
    torch.save(options, options_path)
    torch.save(pos_tensor, pos_path)
    save_jsonl(rows_path, rows_out)
    options_txt_path.write_text(
        json.dumps(unique_options, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    disk_bytes = dir_size_bytes(out_dir)
    meta = {
        "n_rows": len(examples),
        "n_unique_options": len(unique_options),
        "hidden_size": int(states.shape[1]),
        "states_shape": list(states.shape),
        "options_shape": list(options.shape),
        "pool": embed_cfg.pool_mode,
        "prompt_format": PROMPT_FORMAT,
        "dtype": "float16",
        "max_length": max_length,
        "embed_batch_size": embed_batch_size,
        "encode_seconds": round(encode_s, 2),
        "disk_bytes": disk_bytes,
        "disk_human": format_bytes(disk_bytes),
        "eval_excluded": str(EVAL_PATH.relative_to(ROOT)),
        "files": {
            "states": states_path.name,
            "options": options_path.name,
            "positive_idx": pos_path.name,
            "rows": rows_path.name,
            "option_texts": options_txt_path.name,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"number of train rows to cache (default {DEFAULT_N})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke-test alias: cache only this many rows (overrides --n)",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--embed-batch-size",
        type=int,
        default=DEFAULT_EMBED_BATCH_SIZE,
    )
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument(
        "--out",
        type=Path,
        default=CACHE_DIR,
        help="output directory (default: cache/)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n = args.limit if args.limit is not None else args.n
    if n < 1:
        raise SystemExit("--n / --limit must be >= 1")
    if args.embed_batch_size < 1:
        raise SystemExit("--embed-batch-size must be >= 1")

    print(f"Collecting {n} train examples (seed={args.seed}, eval held out)…")
    examples = collect_train_examples(n, seed=args.seed)
    n_choice = sum(1 for e in examples if e["qtype"] == "choice")
    n_noul = sum(1 for e in examples if e["qtype"] == "noul")
    print(f"  got {len(examples)} rows  (choice={n_choice}, noul={n_noul})")

    meta = build_cache(
        examples,
        out_dir=args.out,
        embed_batch_size=args.embed_batch_size,
        max_length=args.max_length,
    )

    print("\n--- cache done ---")
    print(f"  states:   {meta['states_shape']}")
    print(f"  options:  {meta['options_shape']}  (unique)")
    print(f"  encode:   {meta['encode_seconds']} s")
    print(f"  disk:     {meta['disk_human']}  ({meta['disk_bytes']} bytes)")
    print(f"  wrote:    {args.out}/")


if __name__ == "__main__":
    main()
