"""Load bev-decision and normalize into a common decision format.

Each example:
  state        — context / situation
  question     — what to decide
  choices      — list of option strings
  correct_idx  — index of the right choice
  qtype        — "choice" | "noul"
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EVAL_PATH = DATA_DIR / "eval_bev.jsonl"
EVAL_SIZE = 200
SEED = 42

# noul is yes/no; True → yes, False → no
NOUL_CHOICES = ["yes", "no"]


def _choice_example(state: str, q: dict[str, Any], domain: str) -> dict[str, Any] | None:
    criteria = q.get("criteria") or {}
    label = q.get("label")
    if not criteria or label not in criteria:
        return None
    keys = list(criteria.keys())
    choices = [criteria[k] for k in keys]
    return {
        "state": state,
        "question": q.get("instructions", ""),
        "choices": choices,
        "correct_idx": keys.index(label),
        "qtype": "choice",
        "domain": domain,
    }


def _noul_example(state: str, q: dict[str, Any], domain: str) -> dict[str, Any] | None:
    label = q.get("label")
    if not isinstance(label, bool):
        return None
    return {
        "state": state,
        "question": q.get("instructions", ""),
        "choices": list(NOUL_CHOICES),
        "correct_idx": 0 if label else 1,
        "qtype": "noul",
        "domain": domain,
    }


def row_to_examples(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one HF row (possibly many questions) into decision examples."""
    state = row["state"]
    domain = row.get("domain", "")
    questions = json.loads(row["questions_json"])
    out: list[dict[str, Any]] = []
    for q in questions.values():
        qtype = q.get("type")
        if qtype == "choice":
            ex = _choice_example(state, q, domain)
        elif qtype == "noul":
            ex = _noul_example(state, q, domain)
        else:
            continue  # skip score / unknown for now
        if ex is not None:
            out.append(ex)
    return out


def iter_bev_examples(split: str = "train") -> Iterator[dict[str, Any]]:
    ds = load_dataset("avbiswas/bev-decision", split=split)
    for row in ds:
        yield from row_to_examples(row)


def load_bev_examples(split: str = "train", limit: int | None = None) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for ex in iter_bev_examples(split=split):
        examples.append(ex)
        if limit is not None and len(examples) >= limit:
            break
    return examples


def make_eval_split(
    examples: list[dict[str, Any]],
    n: int = EVAL_SIZE,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Fixed seeded sample — never train on these rows."""
    if len(examples) < n:
        raise ValueError(f"need at least {n} examples, got {len(examples)}")
    rng = random.Random(seed)
    return rng.sample(examples, n)


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    print("Loading avbiswas/bev-decision (train)…")
    examples = load_bev_examples(split="train")
    n_choice = sum(1 for e in examples if e["qtype"] == "choice")
    n_noul = sum(1 for e in examples if e["qtype"] == "noul")
    print(f"converted examples: {len(examples):,}  (choice={n_choice:,}, noul={n_noul:,})")

    print("\n--- sample converted examples ---")
    for i, ex in enumerate(examples[:3]):
        print(f"\n=== example {i} ({ex['qtype']}) ===")
        print(f"state[:160]: {ex['state'][:160]!r}")
        print(f"question:     {ex['question']}")
        print(f"choices:      {ex['choices']}")
        print(f"correct_idx:  {ex['correct_idx']} → {ex['choices'][ex['correct_idx']]}")

    eval_rows = make_eval_split(examples, n=EVAL_SIZE, seed=SEED)
    save_jsonl(EVAL_PATH, eval_rows)
    print(f"\nWrote {len(eval_rows)} seeded eval rows → {EVAL_PATH}")
    print(f"  choice={sum(1 for e in eval_rows if e['qtype'] == 'choice')}")
    print(f"  noul={sum(1 for e in eval_rows if e['qtype'] == 'noul')}")


if __name__ == "__main__":
    main()
