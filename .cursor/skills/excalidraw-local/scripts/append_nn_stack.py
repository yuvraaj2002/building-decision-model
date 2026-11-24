#!/usr/bin/env python3
"""Append a neural-net-feel layer stack with vertical dimension bars.

Each --layer is `label:dim` (e.g. Linear:1024). Bar height scales with dim
so 1024 looks tall and 512 looks half as tall — like a tensor block.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _excalidraw import (
    active_elements,
    bounds,
    load,
    make_arrow,
    make_rect,
    make_separator,
    make_text,
    save,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--title", required=True)
    ap.add_argument(
        "--layers",
        required=True,
        help="Pipe-separated label:dim items, e.g. 'cached:1024|Linear:512|LN:512'",
    )
    ap.add_argument("--note", default="")
    ap.add_argument("--bar-width", type=float, default=56)
    ap.add_argument("--px-per-dim", type=float, default=0.22, help="bar height = dim * px")
    ap.add_argument("--min-bar-h", type=float, default=48)
    ap.add_argument("--gap", type=float, default=70)
    args = ap.parse_args()

    path = Path(args.path)
    data = load(path)
    els = active_elements(data)
    b = bounds(els)
    base_x = 100.0
    if b:
        base_y = b[3] + 80.0
        sep_width = max(900.0, b[2] - b[0] + 40.0)
    else:
        base_y = 60.0
        sep_width = 1000.0

    layers: list[tuple[str, float]] = []
    for part in args.layers.split("|"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            print(f"bad layer (want label:dim): {part!r}", file=sys.stderr)
            return 1
        label, dim_s = part.rsplit(":", 1)
        layers.append((label.strip().replace("\\n", "\n"), float(dim_s)))
    if not layers:
        print("No layers", file=sys.stderr)
        return 1

    claimed: set[str] = {e.get("index") for e in data.get("elements", []) if e.get("index")}

    def idx() -> str:
        n = 0
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        while True:
            cand = f"z{chars[n]}" if n < len(chars) else f"z{n}"
            n += 1
            if cand not in claimed:
                claimed.add(cand)
                return cand

    new_els = []
    if els:
        new_els.append(make_separator(base_x - 20, base_y - 40, sep_width, idx()))

    new_els.append(make_text(base_x, base_y, args.title, idx(), font_size=28))

    max_h = max(max(args.min_bar_h, d * args.px_per_dim) for _, d in layers)
    stack_top = base_y + 70
    baseline_y = stack_top + max_h  # bars grow upward from a shared baseline

    # Dark-theme muted fills (readable on #121212)
    colors = ["#1c4966", "#5c2a2a", "#5c5320", "#1e4d2b", "#5c2a2a", "#1a3344", "#3d2a5c"]
    centers: list[tuple[float, float]] = []

    for i, (label, dim) in enumerate(layers):
        h = max(args.min_bar_h, dim * args.px_per_dim)
        x = base_x + i * (args.bar_width + args.gap)
        y = baseline_y - h
        bg = colors[i % len(colors)]
        bar = make_rect(x, y, args.bar_width, h, idx(), bg=bg)
        # dim label inside / on bar
        dim_label = make_text(
            x + 6,
            y + max(8.0, h / 2 - 18),
            str(int(dim)) if dim == int(dim) else str(dim),
            idx(),
            font_size=16,
        )
        # layer name under baseline
        name = make_text(x - 10, baseline_y + 16, label, idx(), font_size=14)
        new_els.extend([bar, dim_label, name])
        centers.append((x + args.bar_width / 2, y + h / 2))

    for i in range(len(centers) - 1):
        x1, y1 = centers[i]
        x2, y2 = centers[i + 1]
        # arrow from right edge of bar i toward left of bar i+1
        x1 = base_x + i * (args.bar_width + args.gap) + args.bar_width + 4
        x2 = base_x + (i + 1) * (args.bar_width + args.gap) - 4
        new_els.append(make_arrow(x1, y1, x2, y2, idx()))

    note_y = baseline_y + 70
    if args.note:
        new_els.append(make_text(base_x, note_y, args.note.replace("\\n", "\n"), idx(), font_size=16))

    data.setdefault("elements", []).extend(new_els)
    save(path, data)
    print(f"appended {len(layers)} dim-bars → {path} ({len(new_els)} elements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
