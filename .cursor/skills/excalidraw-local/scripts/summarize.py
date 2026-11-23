#!/usr/bin/env python3
"""Print a compact summary of an Excalidraw file. Never prints embedded file bytes."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    path = Path(args.path)
    if not path.exists():
        print(f"MISSING {path}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text())
    els = [e for e in data.get("elements", []) if not e.get("isDeleted")]
    files = data.get("files") or {}

    types = Counter(e.get("type") for e in els)
    print(f"file: {path}")
    print(f"size_bytes: {path.stat().st_size}")
    print(f"elements: {len(els)}  deleted_hidden: {len(data.get('elements', [])) - len(els)}")
    print(f"types: {dict(types)}")
    print(f"embedded_files: {len(files)}")

    if els:
        xs, ys = [], []
        for e in els:
            x, y = e.get("x", 0), e.get("y", 0)
            w, h = e.get("width", 0), e.get("height", 0)
            xs += [x, x + w]
            ys += [y, y + h]
        print(f"bounds: x=[{min(xs):.0f},{max(xs):.0f}] y=[{min(ys):.0f},{max(ys):.0f}]")

    texts = [e.get("text", "") for e in els if e.get("type") == "text" and e.get("text")]
    print(f"texts ({len(texts)}):")
    for t in texts:
        one = " / ".join(line.strip() for line in t.splitlines() if line.strip())
        print(f"  - {one}")

    for e in els:
        if e.get("type") == "image":
            print(
                f"image: id={e.get('id')} status={e.get('status')} "
                f"fileId={e.get('fileId')} {e.get('width')}x{e.get('height')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
