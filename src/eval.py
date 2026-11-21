"""Zero-shot decision eval: closest-choice cosine on frozen embeddings.

For each example:
  query  = embed(state + question)
  scores = [cosine(query, embed(choice_i)) for each choice]
  pred   = argmax(scores)
Compare pred to correct_idx → top-1 accuracy vs random chance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data import EVAL_PATH, load_jsonl
from embed import POOL_MODE, embed_texts

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def query_text(example: dict) -> str:
    """Join situation + question into one string to embed."""
    return f"{example['state']}\n\n{example['question']}"


def decide(
    example: dict,
    *,
    mode: str = POOL_MODE,
) -> tuple[int, torch.Tensor]:
    """Embed query + choices; return (pred_idx, scores[num_choices])."""
    texts = [query_text(example), *example["choices"]]
    vecs = embed_texts(texts, mode=mode)  # [1 + C, H]
    query = vecs[0:1]  # [1, H]
    choices = vecs[1:]  # [C, H]
    scores = F.cosine_similarity(query, choices)  # [C]
    pred_idx = int(scores.argmax().item())
    return pred_idx, scores.detach().cpu()


def random_chance(examples: list[dict]) -> float:
    """Expected accuracy if you pick uniformly among each row's choices."""
    if not examples:
        return 0.0
    return sum(1.0 / len(ex["choices"]) for ex in examples) / len(examples)


def evaluate(
    examples: list[dict],
    *,
    mode: str = POOL_MODE,
    progress_every: int = 10,
) -> dict:
    """Run closest-choice eval; return accuracy stats."""
    n = len(examples)
    correct = 0
    by_qtype: dict[str, list[bool]] = {}

    for i, ex in enumerate(examples):
        pred_idx, _scores = decide(ex, mode=mode)
        hit = pred_idx == ex["correct_idx"]
        correct += int(hit)
        by_qtype.setdefault(ex.get("qtype", "?"), []).append(hit)

        if progress_every and (i + 1) % progress_every == 0:
            so_far = correct / (i + 1)
            print(f"  [{i + 1}/{n}] running accuracy={so_far:.3f}")

    accuracy = correct / n if n else 0.0
    chance = random_chance(examples)
    qtype_acc = {
        qt: sum(hits) / len(hits) for qt, hits in by_qtype.items() if hits
    }

    return {
        "n": n,
        "correct": correct,
        "accuracy": accuracy,
        "random_chance": chance,
        "pool_mode": mode,
        "by_qtype": qtype_acc,
    }


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
        "--save",
        action="store_true",
        help="write results/day5_eval.json",
    )
    args = parser.parse_args()

    examples = load_jsonl(args.path)
    if args.limit is not None:
        examples = examples[: args.limit]

    print(f"eval path: {args.path}")
    print(f"n={len(examples)}  pool={args.mode}")
    print("scoring: argmax cosine(embed(state+question), embed(choice_i))")

    stats = evaluate(examples, mode=args.mode)
    print("\n--- results ---")
    print(f"top-1 accuracy:  {stats['accuracy']:.4f}  ({stats['correct']}/{stats['n']})")
    print(f"random chance:   {stats['random_chance']:.4f}")
    print(f"lift over chance:{stats['accuracy'] - stats['random_chance']:+.4f}")
    for qt, acc in sorted(stats["by_qtype"].items()):
        print(f"  {qt}: {acc:.4f}")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "day5_eval.json"
        out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
