#!/usr/bin/env python3
"""Append a titled horizontal flow diagram below existing Excalidraw content."""
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
        "--steps",
        required=True,
        help="Pipe-separated steps; use \\n for line breaks inside a step",
    )
    ap.add_argument("--note", default="")
    ap.add_argument("--detail", default="", help="Detail box under the flow")
    ap.add_argument("--box-width", type=float, default=200)
    ap.add_argument("--box-height", type=float, default=120)
    ap.add_argument("--gap", type=float, default=40)
    args = ap.parse_args()

    path = Path(args.path)
    data = load(path)
    els = active_elements(data)
    b = bounds(els)
    base_x = 100.0
    if b:
        base_y = b[3] + 80.0
        sep_width = max(800.0, b[2] - b[0] + 40.0)
    else:
        base_y = 60.0
        sep_width = 1000.0

    steps = [s.strip() for s in args.steps.split("|") if s.strip()]
    if not steps:
        print("No steps provided", file=sys.stderr)
        return 1

    new_els = []
    claimed: set[str] = {e.get("index") for e in data.get("elements", []) if e.get("index")}

    def idx() -> str:
        # allocate uniquely against file + this batch
        n = 0
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        while True:
            cand = f"z{chars[n]}" if n < len(chars) else f"z{n}"
            n += 1
            if cand not in claimed:
                claimed.add(cand)
                return cand

    if els:
        new_els.append(make_separator(base_x - 20, base_y - 40, sep_width, idx()))

    new_els.append(make_text(base_x, base_y, args.title, idx(), font_size=28))

    box_y = base_y + 70
    box_w, box_h, gap = args.box_width, args.box_height, args.gap
    rects = []
    for i, step in enumerate(steps):
        x = base_x + i * (box_w + gap)
        r = make_rect(x, box_y, box_w, box_h, idx(), bg="#fff8e7")
        t = make_text(x + 12, box_y + 18, step, idx(), font_size=16)
        new_els.extend([r, t])
        rects.append(r)

    for i in range(len(rects) - 1):
        a, b2 = rects[i], rects[i + 1]
        x1 = a["x"] + a["width"]
        y1 = a["y"] + a["height"] / 2
        x2 = b2["x"]
        y2 = b2["y"] + b2["height"] / 2
        new_els.append(make_arrow(x1, y1, x2, y2, idx()))

    detail_y = box_y + box_h + 50
    if args.detail:
        cx = base_x + max(0, (len(steps) // 2) - 1) * (box_w + gap)
        new_els.append(make_rect(cx, detail_y, 320, 160, idx(), bg="#e8f4ff"))
        new_els.append(make_text(cx + 14, detail_y + 16, args.detail, idx(), font_size=16))

    if args.note:
        new_els.append(make_text(base_x, detail_y + 20, args.note, idx(), font_size=16))

    data.setdefault("elements", []).extend(new_els)
    save(path, data)
    print(f"appended {len(steps)} steps → {path} ({len(new_els)} elements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
