"""Day 10: training loop for BiEncoderHeads on cached frozen Qwen vectors.

Reads Day 7 cache (`cache/states.pt`, `cache/options.pt`, `cache/positive_idx.pt`)
and trains `BiEncoderHeads` with `infonce_loss` on MPS (or CPU fallback).

Examples:
  # Day 10 sanity check: overfit 32 pairs to ~0 loss
  uv run python src/train.py --overfit 32

  # Full training on cached 5k pairs (batch 64)
  uv run python src/train.py --epochs 10 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from data import EVAL_PATH, load_jsonl
from heads import BiEncoderHeads, DEFAULT_OUT_DIM
from loss import infonce_loss

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "cache"
DEFAULT_EVAL_CACHE = DEFAULT_CACHE_DIR / "eval_bev.pt"
DEFAULT_LOG_JSON = ROOT / "results" / "m2_run1.json"


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def load_cache(cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load cached float32 state vectors, option vectors, and positive indices."""
    states = torch.load(cache_dir / "states.pt", map_location="cpu", weights_only=True).float()
    options = torch.load(cache_dir / "options.pt", map_location="cpu", weights_only=True).float()
    pos_idx = torch.load(cache_dir / "positive_idx.pt", map_location="cpu", weights_only=True).long()
    if states.shape[0] != pos_idx.shape[0]:
        raise ValueError(f"states ({states.shape[0]}) != positive_idx ({pos_idx.shape[0]})")
    return states, options, pos_idx


def select_unique_positive_rows(pos_idx: torch.Tensor, k: int) -> torch.Tensor:
    """Pick k row indices whose positive_idx values are all distinct.

    Avoids duplicate positive options (e.g. repeated 'yes'/'no' noul options)
    so the overfit-k sanity check has an achievable ~0 loss floor.
    """
    seen: set[int] = set()
    chosen: list[int] = []
    for row_i, opt_i in enumerate(pos_idx.tolist()):
        if opt_i not in seen:
            seen.add(opt_i)
            chosen.append(row_i)
            if len(chosen) == k:
                break
    if len(chosen) < k:
        raise ValueError(f"only found {len(chosen)} unique positive options, need {k}")
    return torch.tensor(chosen, dtype=torch.long)


def in_batch_top1(scores: torch.Tensor) -> float:
    preds = scores.argmax(dim=-1)
    targets = torch.arange(scores.shape[0], device=scores.device)
    return (preds == targets).float().mean().item()


def load_or_build_eval_cache(eval_cache_path: Path) -> dict:
    """Load cached 200-row eval embeddings, or encode once with frozen Qwen."""
    if eval_cache_path.exists():
        return torch.load(eval_cache_path, map_location="cpu", weights_only=False)

    from embed import POOL_MODE, embed_texts, sync
    from eval import (
        DEFAULT_EMBED_BATCH_SIZE,
        DEFAULT_MAX_LENGTH,
        format_choice,
        format_query,
    )

    examples = load_jsonl(EVAL_PATH)
    query_texts = [format_query(ex, "labeled") for ex in examples]
    unique_options: list[str] = []
    opt_to_id: dict[str, int] = {}
    choice_option_ids: list[list[int]] = []
    correct_idx: list[int] = []
    qtypes: list[str] = []

    for ex in examples:
        row_ids: list[int] = []
        for c in ex["choices"]:
            txt = format_choice(c, "labeled")
            oid = opt_to_id.get(txt)
            if oid is None:
                oid = len(unique_options)
                opt_to_id[txt] = oid
                unique_options.append(txt)
            row_ids.append(oid)
        choice_option_ids.append(row_ids)
        correct_idx.append(int(ex["correct_idx"]))
        qtypes.append(ex.get("qtype", "unknown"))

    print(
        f"Building eval cache ({len(query_texts)} states, {len(unique_options)} unique options; "
        f"labeled, pool={POOL_MODE}, fp16, max_length={DEFAULT_MAX_LENGTH})..."
    )

    def _encode_all(texts: list[str]) -> torch.Tensor:
        chunks = []
        for i in range(0, len(texts), DEFAULT_EMBED_BATCH_SIZE):
            vecs = embed_texts(
                texts[i : i + DEFAULT_EMBED_BATCH_SIZE],
                mode=POOL_MODE,
                dtype=torch.float16,
                max_length=DEFAULT_MAX_LENGTH,
            )
            chunks.append(vecs.detach().cpu().to(torch.float16))
        return torch.cat(chunks, dim=0)

    sync()
    t0 = time.monotonic()
    states = _encode_all(query_texts)
    options = _encode_all(unique_options)
    sync()
    print(f"Saved eval cache to {eval_cache_path} in {time.monotonic() - t0:.1f}s")

    payload = {
        "states": states,
        "options": options,
        "choice_option_ids": choice_option_ids,
        "correct_idx": correct_idx,
        "qtypes": qtypes,
        "prompt_format": "labeled",
        "pool": POOL_MODE,
        "max_length": DEFAULT_MAX_LENGTH,
    }
    eval_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, eval_cache_path)
    return payload


@torch.no_grad()
def evaluate_heads(
    model: BiEncoderHeads,
    eval_data: dict,
    device: torch.device,
) -> dict[str, float]:
    """Score every eval row against its own candidate options → top-1 accuracy."""
    model.eval()
    s_emb = model.encode_states(eval_data["states"].float().to(device))
    a_emb = model.encode_actions(eval_data["options"].float().to(device))

    correct_all = 0
    correct_by_type: dict[str, int] = {}
    total_by_type: dict[str, int] = {}
    chance_sum = 0.0

    for i, (opt_ids, gold, qtype) in enumerate(
        zip(eval_data["choice_option_ids"], eval_data["correct_idx"], eval_data["qtypes"])
    ):
        cand = a_emb[opt_ids]  # [K, D]
        scores = cand @ s_emb[i]  # [K] (scale > 0 doesn't change argmax)
        pred = int(scores.argmax().item())
        hit = int(pred == gold)
        correct_all += hit
        correct_by_type[qtype] = correct_by_type.get(qtype, 0) + hit
        total_by_type[qtype] = total_by_type.get(qtype, 0) + 1
        chance_sum += 1.0 / len(opt_ids)

    n = len(eval_data["correct_idx"])
    out = {
        "top1": correct_all / n,
        "chance": chance_sum / n,
        "n": n,
    }
    for qt, tot in sorted(total_by_type.items()):
        out[f"top1_{qt}"] = correct_by_type[qt] / tot
    return out


def run_overfit(
    states: torch.Tensor,
    options: torch.Tensor,
    pos_idx: torch.Tensor,
    *,
    k: int,
    steps: int,
    lr: float,
    weight_decay: float,
    out_dim: int,
    symmetric: bool,
    device: torch.device,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    row_ids = select_unique_positive_rows(pos_idx, k)
    s_batch = states[row_ids].to(device)
    a_batch = options[pos_idx[row_ids]].to(device)

    model = BiEncoderHeads(in_dim=states.shape[1], out_dim=out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(
        f"Overfit sanity check: k={k} unique pairs | steps={steps} | "
        f"lr={lr} | device={device} | log(k)={math.log(k):.4f}"
    )

    sync_device(device)
    t0 = time.monotonic()
    first_loss: float | None = None
    last_loss = 0.0
    last_acc = 0.0

    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        scores = model.score_matrix(s_batch, a_batch)
        loss = infonce_loss(scores, symmetric=symmetric)
        loss.backward()
        opt.step()

        loss_val = loss.item()
        acc_val = in_batch_top1(scores)
        if first_loss is None:
            first_loss = loss_val
        last_loss, last_acc = loss_val, acc_val

        if step == 1 or step % 20 == 0 or step == steps:
            print(
                f"  step {step:4d}/{steps} | loss {loss_val:.5f} | "
                f"in-batch top1 {acc_val * 100:5.1f}% | scale {model.scale.item():.2f}"
            )

    sync_device(device)
    elapsed = time.monotonic() - t0
    print(
        f"Done in {elapsed:.2f}s | loss {first_loss:.4f} -> {last_loss:.5f} | "
        f"in-batch top1 {last_acc * 100:.1f}%"
    )


def run_train(
    states: torch.Tensor,
    options: torch.Tensor,
    pos_idx: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    out_dim: int,
    symmetric: bool,
    device: torch.device,
    seed: int,
    eval_cache_path: Path,
    log_json_path: Path | None,
    save_path: Path | None,
) -> None:
    torch.manual_seed(seed)
    eval_data = load_or_build_eval_cache(eval_cache_path)

    pos_actions = options[pos_idx]
    ds = TensorDataset(states, pos_actions)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = BiEncoderHeads(in_dim=states.shape[1], out_dim=out_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(
        f"Training on {len(ds)} pairs | epochs={epochs} | batch_size={batch_size} "
        f"({len(loader)} steps/epoch) | lr={lr} | device={device}"
    )

    history: list[dict] = []
    best_top1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    sync_device(device)
    t0 = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            opt.zero_grad()
            scores = model.score_matrix(s_batch, a_batch)
            loss = infonce_loss(scores, symmetric=symmetric)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_acc += in_batch_top1(scores)

        avg_loss = total_loss / len(loader)
        avg_acc = total_acc / len(loader)
        ev = evaluate_heads(model, eval_data, device)

        row = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 5),
            "in_batch_top1": round(avg_acc, 4),
            "scale": round(model.scale.item(), 4),
            "eval_top1": round(ev["top1"], 4),
            "eval_choice": round(ev.get("top1_choice", 0.0), 4),
            "eval_noul": round(ev.get("top1_noul", 0.0), 4),
        }
        history.append(row)

        marker = ""
        if ev["top1"] > best_top1:
            best_top1 = ev["top1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"

        print(
            f"  epoch {epoch:3d}/{epochs} | loss {avg_loss:.4f} | "
            f"in-batch {avg_acc * 100:5.1f}% | "
            f"eval top1 {ev['top1'] * 100:5.1f}% "
            f"(choice {ev.get('top1_choice', 0.0) * 100:4.1f}%, "
            f"noul {ev.get('top1_noul', 0.0) * 100:4.1f}%) | "
            f"scale {model.scale.item():.2f}{marker}"
        )

    sync_device(device)
    elapsed = time.monotonic() - t0
    print(
        f"Training finished in {elapsed:.2f}s | "
        f"best eval top1: {best_top1 * 100:.1f}% (epoch {best_epoch}) "
        f"vs M1 baseline 52.5%"
    )

    if log_json_path is not None:
        log_json_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "n_train": len(ds),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "out_dim": out_dim,
            "symmetric": symmetric,
            "seed": seed,
            "device": str(device),
            "elapsed_seconds": round(elapsed, 2),
            "best_epoch": best_epoch,
            "best_eval_top1": round(best_top1, 4),
            "m1_baseline_top1": 0.525,
            "history": history,
        }
        log_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved training log to {log_json_path}")

    if save_path is not None and best_state is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "in_dim": states.shape[1],
                "out_dim": out_dim,
                "best_epoch": best_epoch,
                "best_eval_top1": best_top1,
                "batch_size": batch_size,
                "lr": lr,
                "symmetric": symmetric,
            },
            save_path,
        )
        print(f"Saved best checkpoint to {save_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--eval-cache", type=Path, default=DEFAULT_EVAL_CACHE)
    p.add_argument("--overfit", type=int, default=0, help="overfit k unique pairs (Day 10 sanity check)")
    p.add_argument("--steps", type=int, default=200, help="steps for --overfit mode")
    p.add_argument("--epochs", type=int, default=10, help="epochs for full training mode")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--out-dim", type=int, default=DEFAULT_OUT_DIM)
    p.add_argument("--symmetric", action="store_true", help="use CLIP-style symmetric InfoNCE")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-json", type=Path, default=DEFAULT_LOG_JSON, help="eval/loss history JSON path")
    p.add_argument("--save", type=Path, default=None, help="optional best-checkpoint path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    states, options, pos_idx = load_cache(args.cache_dir)

    if args.overfit > 0:
        run_overfit(
            states,
            options,
            pos_idx,
            k=args.overfit,
            steps=args.steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            out_dim=args.out_dim,
            symmetric=args.symmetric,
            device=device,
            seed=args.seed,
        )
    else:
        run_train(
            states,
            options,
            pos_idx,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            out_dim=args.out_dim,
            symmetric=args.symmetric,
            device=device,
            seed=args.seed,
            eval_cache_path=args.eval_cache,
            log_json_path=args.log_json,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()
