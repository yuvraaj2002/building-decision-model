#!/usr/bin/env python3
"""Append a text note below existing Excalidraw content."""
from __future__ import annotations

import argparse
from pathlib import Path

from _excalidraw import active_elements, bounds, load, make_text, next_index, save


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--text", required=True)
    ap.add_argument("--font-size", type=int, default=16)
    args = ap.parse_args()

    path = Path(args.path)
    data = load(path)
    els = active_elements(data)
    b = bounds(els)
    x, y = (100.0, 60.0) if not b else (b[0], b[3] + 60.0)
    idx = next_index(data)
    data.setdefault("elements", []).append(
        make_text(x, y, args.text, idx, font_size=args.font_size)
    )
    save(path, data)
    print(f"note appended → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
