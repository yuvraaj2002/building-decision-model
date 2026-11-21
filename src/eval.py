"""Zero-shot decision eval: closest-choice cosine on frozen embeddings.

For each example:
  query  = embed(state + question)
  scores = [cosine(query, embed(choice_i)) for each choice]
  pred   = argmax(scores)
Compare pred to correct_idx → top-1 accuracy vs random chance.

Fast defaults for Mac 24GB: truncate to 256 tokens, fp16, batch rows.
Reports amortized ms/decision plus p50 / p95.

Prompt formats:
  plain   — raw state + question; raw choice strings
  labeled — "Situation:" / "Question:" query; "Option:" choices
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import EVAL_PATH, load_jsonl
from embed import POOL_MODE, embed_texts, sync

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

# Fast path: truncate long states so batching is safe on 24GB unified memory.
DEFAULT_ROWS_PER_BATCH = 8
DEFAULT_EMBED_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 256
PROMPT_FORMATS = ("plain", "labeled")


def format_query(example: dict, prompt_format: str) -> str:
    """Build the query string for one prompt format."""
    state = example["state"]
    question = example["question"]
    if prompt_format == "plain":
        return f"{state}\n\n{question}"
    if prompt_format == "labeled":
        return f"Situation: {state}\nQuestion: {question}"
    raise ValueError(f"unknown prompt format: {prompt_format!r}")


def format_choice(choice: str, prompt_format: str) -> str:
    """Build one choice string for one prompt format."""
    if prompt_format == "plain":
        return choice
    if prompt_format == "labeled":
        return f"Option: {choice}"
    raise ValueError(f"unknown prompt format: {prompt_format!r}")


def format_choice_list(choices: list[str], prompt_format: str) -> list[str]:
    return [format_choice(c, prompt_format) for c in choices]


def embed_many(
    texts: list[str],
    *,
    mode: str,
    embed_batch_size: int,
    dtype: torch.dtype,
    max_length: int | None,
) -> torch.Tensor:
    """Embed texts in GPU-sized chunks; concat → [N, H]."""
    if not texts:
        raise ValueError("embed_many got an empty text list")
    parts: list[torch.Tensor] = []
    for i in range(0, len(texts), embed_batch_size):
        chunk = texts[i : i + embed_batch_size]
        parts.append(
            embed_texts(chunk, mode=mode, dtype=dtype, max_length=max_length)
        )
    return torch.cat(parts, dim=0)


def decide(
    example: dict,
    *,
    mode: str = POOL_MODE,
    prompt_format: str = "plain",
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    dtype: torch.dtype = torch.float16,
    max_length: int | None = DEFAULT_MAX_LENGTH,
) -> tuple[int, torch.Tensor]:
    """Embed query + choices; return (pred_idx, scores[num_choices])."""
    texts = [
        format_query(example, prompt_format),
        *format_choice_list(example["choices"], prompt_format),
    ]
    vecs = embed_many(
        texts,
        mode=mode,
        embed_batch_size=embed_batch_size,
        dtype=dtype,
        max_length=max_length,
    )
    query = vecs[0:1]
    choices = vecs[1:]
    scores = F.cosine_similarity(query.float(), choices.float())
    pred_idx = int(scores.argmax().item())
    return pred_idx, scores.detach().cpu()


def score_chunk(
    examples: list[dict],
    *,
    mode: str,
    prompt_format: str,
    embed_batch_size: int,
    dtype: torch.dtype,
    max_length: int | None,
) -> list[tuple[int, torch.Tensor]]:
    """Batched embed for a list of rows; return (pred_idx, scores) each."""
    texts: list[str] = []
    spans: list[tuple[int, int, int]] = []  # query_idx, choice_start, choice_end

    for ex in examples:
        q_idx = len(texts)
        texts.append(format_query(ex, prompt_format))
        c_start = len(texts)
        texts.extend(format_choice_list(ex["choices"], prompt_format))
        spans.append((q_idx, c_start, len(texts)))

    vecs = embed_many(
        texts,
        mode=mode,
        embed_batch_size=embed_batch_size,
        dtype=dtype,
        max_length=max_length,
    )

    out: list[tuple[int, torch.Tensor]] = []
    for q_idx, c_start, c_end in spans:
        query = vecs[q_idx : q_idx + 1]
        choices = vecs[c_start:c_end]
        scores = F.cosine_similarity(query.float(), choices.float())
        pred_idx = int(scores.argmax().item())
        out.append((pred_idx, scores.detach().cpu()))
    return out


def random_chance(examples: list[dict]) -> float:
    """Expected accuracy if you pick uniformly among each row's choices."""
    if not examples:
        return 0.0
    return sum(1.0 / len(ex["choices"]) for ex in examples) / len(examples)


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile; `values` need not be sorted."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (p / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def evaluate(
    examples: list[dict],
    *,
    mode: str = POOL_MODE,
    prompt_format: str = "plain",
    rows_per_batch: int = DEFAULT_ROWS_PER_BATCH,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    dtype: torch.dtype = torch.float16,
    max_length: int | None = DEFAULT_MAX_LENGTH,
    progress_every: int = 16,
    warmup: bool = True,
) -> dict:
    """Run closest-choice eval; return accuracy + amortized latency stats."""
    n = len(examples)
    correct = 0
    by_qtype: dict[str, list[bool]] = {}
    latencies_ms: list[float] = []

    if warmup and examples:
        _ = decide(
            examples[0],
            mode=mode,
            prompt_format=prompt_format,
            embed_batch_size=embed_batch_size,
            dtype=dtype,
            max_length=max_length,
        )
        sync()

    for start in range(0, n, rows_per_batch):
        chunk = examples[start : start + rows_per_batch]

        sync()
        t0 = time.monotonic()
        preds = score_chunk(
            chunk,
            mode=mode,
            prompt_format=prompt_format,
            embed_batch_size=embed_batch_size,
            dtype=dtype,
            max_length=max_length,
        )
        sync()
        batch_ms = (time.monotonic() - t0) * 1000.0
        amortized_ms = batch_ms / len(chunk)

        for ex, (pred_idx, _scores) in zip(chunk, preds):
            latencies_ms.append(amortized_ms)
            hit = pred_idx == ex["correct_idx"]
            correct += int(hit)
            by_qtype.setdefault(ex.get("qtype", "?"), []).append(hit)

        del preds
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        done = start + len(chunk)
        if progress_every and (
            done % progress_every == 0 or done == n or len(chunk) >= progress_every
        ):
            so_far = correct / done
            print(
                f"  [{done}/{n}] format={prompt_format}  accuracy={so_far:.3f}  "
                f"batch={len(chunk)}  amortized_p50={percentile(latencies_ms, 50):.1f} ms",
                flush=True,
            )

    accuracy = correct / n if n else 0.0
    chance = random_chance(examples)
    qtype_acc = {
        qt: sum(hits) / len(hits) for qt, hits in by_qtype.items() if hits
    }
    mean_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    return {
        "n": n,
        "correct": correct,
        "accuracy": accuracy,
        "random_chance": chance,
        "pool_mode": mode,
        "prompt_format": prompt_format,
        "dtype": "fp16" if dtype == torch.float16 else "fp32",
        "max_length": max_length,
        "by_qtype": qtype_acc,
        "rows_per_batch": rows_per_batch,
        "embed_batch_size": embed_batch_size,
        "latency_ms": {
            "mean": mean_ms,
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "note": "amortized = batch_wall_ms / rows_in_batch",
        },
    }


def print_stats(stats: dict) -> None:
    lat = stats["latency_ms"]
    print("\n--- results ---")
    print(f"prompt format:   {stats['prompt_format']}")
    print(f"pool mode:       {stats['pool_mode']}")
    print(f"dtype/max_len:   {stats['dtype']} / {stats['max_length']}")
    print(f"top-1 accuracy:  {stats['accuracy']:.4f}  ({stats['correct']}/{stats['n']})")
    print(f"random chance:   {stats['random_chance']:.4f}")
    print(f"lift over chance:{stats['accuracy'] - stats['random_chance']:+.4f}")
    for qt, acc in sorted(stats["by_qtype"].items()):
        print(f"  {qt}: {acc:.4f}")
    print(
        f"latency ms/decision (amortized):  mean={lat['mean']:.1f}  "
        f"p50={lat['p50']:.1f}  p95={lat['p95']:.1f}"
    )


def pick_winner(runs: list[dict]) -> dict:
    """Prefer higher accuracy; break ties with lower p50 latency."""
    return max(
        runs,
        key=lambda s: (s["accuracy"], -s["latency_ms"]["p50"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot closest-choice eval")
    parser.add_argument(
        "--path",
        type=Path,
        default=EVAL_PATH,
        help="jsonl eval split (default: data/eval_bev.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N rows (smoke test)",
    )
    parser.add_argument(
        "--mode",
        choices=["mean", "last_token"],
        default=POOL_MODE,
        help="pooling mode (Day 4 winner: mean)",
    )
    parser.add_argument(
        "--format",
        choices=list(PROMPT_FORMATS),
        default="plain",
        help="prompt format: plain | labeled (Situation:/Option:)",
    )
    parser.add_argument(
        "--compare-formats",
        action="store_true",
        help="run plain and labeled once each; report which wins",
    )
    parser.add_argument(
        "--rows-per-batch",
        type=int,
        default=DEFAULT_ROWS_PER_BATCH,
        help="eval rows per scoring batch (default 8 with truncation)",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=DEFAULT_EMBED_BATCH_SIZE,
        help="texts per GPU forward (default 8)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="truncate tokens to this length (default 256; 0 = no truncate)",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="use fp32 instead of fp16",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="write results/day6_eval.json (or results/baseline.json with --compare-formats)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip one warmup decision before timing",
    )
    args = parser.parse_args()

    if args.rows_per_batch < 1:
        raise SystemExit("--rows-per-batch must be >= 1")
    if args.embed_batch_size < 1:
        raise SystemExit("--embed-batch-size must be >= 1")

    max_length = None if args.max_length == 0 else args.max_length
    dtype = torch.float32 if args.fp32 else torch.float16

    examples = load_jsonl(args.path)
    if args.limit is not None:
        examples = examples[: args.limit]

    formats = list(PROMPT_FORMATS) if args.compare_formats else [args.format]

    print(f"eval path: {args.path}")
    print(
        f"n={len(examples)}  pool={args.mode}  formats={formats}  "
        f"dtype={'fp32' if args.fp32 else 'fp16'}  max_length={max_length}  "
        f"rows_per_batch={args.rows_per_batch}  "
        f"embed_batch_size={args.embed_batch_size}"
    )
    print("scoring: argmax cosine(embed(query), embed(choice_i))")

    runs: list[dict] = []
    for prompt_format in formats:
        print(f"\n=== format={prompt_format} ===")
        stats = evaluate(
            examples,
            mode=args.mode,
            prompt_format=prompt_format,
            rows_per_batch=args.rows_per_batch,
            embed_batch_size=args.embed_batch_size,
            dtype=dtype,
            max_length=max_length,
            warmup=not args.no_warmup,
        )
        print_stats(stats)
        runs.append(stats)

    if args.compare_formats:
        winner = pick_winner(runs)
        print("\n=== winner ===")
        print(
            f"{winner['prompt_format']}  "
            f"accuracy={winner['accuracy']:.4f}  "
            f"p50={winner['latency_ms']['p50']:.1f} ms"
        )
        payload = {
            "pool_mode": args.mode,
            "dtype": "fp32" if args.fp32 else "fp16",
            "max_length": max_length,
            "formats": {s["prompt_format"]: s for s in runs},
            "winner": {
                "prompt_format": winner["prompt_format"],
                "accuracy": winner["accuracy"],
                "random_chance": winner["random_chance"],
                "latency_ms": winner["latency_ms"],
            },
        }
    else:
        payload = runs[0]

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = (
            RESULTS_DIR / "baseline.json"
            if args.compare_formats
            else RESULTS_DIR / "day6_eval.json"
        )
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
